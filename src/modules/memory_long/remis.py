import json
import os
import queue
import re
import signal
import subprocess
import threading
import time
import uuid
from datetime import datetime, timezone

import chromadb
import requests
from sentence_transformers import SentenceTransformer

from config import BASE_DIR
from interfaces import LongTermMemoryInterface

EXTRACTION_PROMPT = """Extract only facts the user explicitly stated that would still be worth knowing weeks from now — their name, job, relationships, preferences, or ongoing plans.

Skip greetings, small talk, questions, and anything about the assistant itself. If nothing qualifies, return an empty list.

Always split distinct facts into separate entries in the list — never combine multiple facts into one sentence, even if they were stated together.

Examples:
Input: hi
Output: {"facts": []}
Input: my name is Alex and I work as a nurse
Output: {"facts": ["User's name is Alex", "User works as a nurse"]}
Input: I'm K5031 and I go to UCL
Output: {"facts": ["User's name is K5031", "User attends UCL"]}
Input: what's your name?
Output: {"facts": []}

Conversation:
{conversation}

Return only valid JSON: {"facts": [...]}"""

CONFLICT_CHECK_PROMPT = """You will see an EXISTING fact and a NEW fact about the user. Decide the relationship between them:

SAME     — they express the same fact, just worded differently (same subject, same value)
CONFLICT — they're about the same specific subject, but the value changed (a correction)
DISTINCT — they're about different subjects or facts entirely, even if superficially similar wording

Answer with exactly one word: SAME, CONFLICT, or DISTINCT.

Examples:
EXISTING: User's name is K5031
NEW: User is called K5031
Answer: SAME

EXISTING: User's name is K5031
NEW: User's name is Alex
Answer: CONFLICT

EXISTING: User works as a teacher
NEW: User works as a software engineer
Answer: CONFLICT

EXISTING: User goes to UCL
NEW: User studies computer science at UCL
Answer: DISTINCT

EXISTING: User has a dog named Rex
NEW: User has a cat named Milo
Answer: DISTINCT

EXISTING: User's name is Alex
NEW: Friend's name is Alex
Answer: DISTINCT

EXISTING: {existing}
NEW: {new}
Answer:"""


def _format_conversation(messages: list[dict]) -> str:
    lines = []
    for m in messages:
        if m["role"] in ("user", "assistant"):
            lines.append(f"{m['role']}: {m['content']}")
    return "\n".join(lines)


def _extract_json_facts(raw_text: str) -> list[str]:
    """Small local models sometimes wrap JSON in extra text — pull it out."""
    match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    if not match:
        return []
    try:
        data = json.loads(match.group(0))
        facts = data.get("facts", [])
        return [f for f in facts if isinstance(f, str) and f.strip()]
    except (json.JSONDecodeError, AttributeError):
        return []


class Module(LongTermMemoryInterface):
    """Remis — reminiscence-based long-term memory. Runs its own dedicated
    extraction model (separate from core, to avoid KV-cache thrashing between
    two very different prompt shapes on a shared server), extracts durable
    facts from conversation, dedupes against existing memories, and recalls
    them by semantic similarity."""

    def __init__(self, model_path, n_gpu_layers, n_ctx=4096,
                 collection_name="k_memory",
                 embed_model_name="Qwen/Qwen3-Embedding-0.6B",
                 dedup_distance: float = 0.05,
                 conflict_band_upper: float = 0.35,
                 port: int = 8081, **kwargs):
        self.name = "Remis"
        self.user_id = "user"
        self.model_path = model_path
        self.port = port
        self.base_url = f"http://localhost:{port}"
        # Chroma's default space is squared L2 (not cosine) unless overridden —
        # lower = more similar, 0 = identical.
        # distance <= dedup_distance                -> near-exact string match, skip
        #   (fast path, no LLM call — catches only true near-duplicates, e.g.
        #   the same fact restated almost verbatim)
        # dedup_distance < d <= conflict_band_upper  -> ambiguous; ask the LLM to
        #   classify as SAME (paraphrase, skip) / CONFLICT (correction, replace) /
        #   DISTINCT (different subject, keep both) — this band is intentionally
        #   wide because raw distance alone can't reliably tell "same fact reworded"
        #   apart from "different fact, same sentence template"
        # distance > conflict_band_upper             -> clearly unrelated, store
        self.dedup_distance = dedup_distance
        self.conflict_band_upper = conflict_band_upper

        print(f"[{self.name}] Extraction model: {model_path}")
        self.server = subprocess.Popen(
            ["python", "-m", "llama_cpp.server",
             "--model", model_path,
             "--port", str(port),
             "--n_gpu_layers", str(n_gpu_layers),
             "--n_ctx", str(n_ctx),
             "--flash_attn", "True",
             "--verbose", "False"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            preexec_fn=os.setsid
        )
        self._wait_for_server()

        self.client = chromadb.PersistentClient(path=os.path.join(BASE_DIR, "data", "chroma"))
        self.collection = self.client.get_or_create_collection(
            collection_name, metadata={"hnsw:space": "cosine"}
        )
        self.embedder = SentenceTransformer(embed_model_name, device="cpu")

        self._queue = queue.Queue()
        self._worker = threading.Thread(target=self._process_queue, daemon=True)
        self._worker.start()

    def _wait_for_server(self, timeout: int = 60):
        start = time.time()
        while time.time() - start < timeout:
            try:
                r = requests.get(f"{self.base_url}/v1/models")
                if r.status_code == 200:
                    return
            except requests.exceptions.ConnectionError:
                pass
            time.sleep(1)
        raise TimeoutError(f"{self.name} llama-server failed to start within timeout")

    def _embed(self, text: str) -> list[float]:
        return self.embedder.encode(text).tolist()

    def _find_nearest(self, embedding: list[float]):
        """Returns (id, text, distance) of the closest existing memory, or None."""
        if self.collection.count() == 0:
            return None
        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=1,
            where={"user_id": self.user_id},
        )
        ids = results.get("ids", [[]])[0]
        distances = results.get("distances", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        if not ids:
            return None
        return ids[0], metadatas[0].get("data", ""), distances[0]

    def _check_conflict(self, existing_text: str, new_text: str) -> str:
        """Ask the model to classify the relationship between an existing
        memory and a new candidate fact. Returns 'SAME', 'CONFLICT', or
        'DISTINCT'. Defaults to 'DISTINCT' on any failure — the safe choice,
        since it never deletes anything, at worst it leaves a redundant entry."""
        prompt = CONFLICT_CHECK_PROMPT.replace("{existing}", existing_text).replace("{new}", new_text)
        try:
            resp = requests.post(
                f"{self.base_url}/v1/chat/completions",
                json={
                    "model": "local-model",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 10,
                    "temperature": 0.0,
                },
                timeout=30,
            )
            resp.raise_for_status()
            raw = resp.json()["choices"][0]["message"]["content"].strip().upper()
            if "SAME" in raw:
                return "SAME"
            if "CONFLICT" in raw:
                return "CONFLICT"
            return "DISTINCT"
        except Exception as e:
            print(f"[{self.name}] conflict check failed, treating as distinct: {e}")
            return "DISTINCT"

    def _process_queue(self):
        while True:
            messages = self._queue.get()
            if messages is None:
                self._queue.task_done()
                break
            try:
                facts = self._extract_and_save(messages)
                if facts:
                    print(f"[{self.name}] stored: {facts}")
            except Exception as e:
                print(f"[{self.name}] store failed: {e}")
            self._queue.task_done()

    def _extract_and_save(self, messages: list[dict]) -> list[str]:
        conversation = _format_conversation(messages)
        if not conversation.strip():
            return []

        prompt = EXTRACTION_PROMPT.replace("{conversation}", conversation)
        resp = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json={
                "model": "local-model",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 300,
                "temperature": 0.0,
            },
            timeout=60,
        )
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"]
        facts = _extract_json_facts(raw)
        if not facts:
            return []

        stored_facts = []
        for f in facts:
            emb = self._embed(f)
            nearest = self._find_nearest(emb)

            if nearest is None:
                self._add_fact(f, emb)
                stored_facts.append(f)
                continue

            nearest_id, nearest_text, distance = nearest

            if distance <= self.dedup_distance:
                print(f"[{self.name}] skipped duplicate: {f}")
                continue

            if distance <= self.conflict_band_upper:
                verdict = self._check_conflict(nearest_text, f)
                if verdict == "SAME":
                    print(f"[{self.name}] skipped duplicate (paraphrase): {f}")
                elif verdict == "CONFLICT":
                    print(f"[{self.name}] conflict detected — replacing "
                          f"'{nearest_text}' with '{f}'")
                    self.collection.delete(ids=[nearest_id])
                    self._add_fact(f, emb)
                    stored_facts.append(f)
                else:  # DISTINCT
                    self._add_fact(f, emb)
                    stored_facts.append(f)
                continue

            self._add_fact(f, emb)
            stored_facts.append(f)

        return stored_facts

    def _add_fact(self, text: str, embedding: list[float]) -> None:
        now = datetime.now(timezone.utc).isoformat()
        self.collection.add(
            ids=[str(uuid.uuid4())],
            embeddings=[embedding],
            metadatas=[{"data": text, "user_id": self.user_id, "created_at": now}],
        )

    # ---- LongTermMemoryInterface ----

    def store(self, messages: list[dict]) -> None:
        self._queue.put(messages)

    def retrieve(self, query: str) -> str:
        if self.collection.count() == 0:
            return ""
        query_embedding = self._embed(query)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(3, self.collection.count()),
            where={"user_id": self.user_id},
        )
        metadatas = results.get("metadatas", [[]])[0]
        facts = [m.get("data", "") for m in metadatas if m.get("data")]
        if not facts:
            return ""
        return "\n".join(f"- {f}" for f in facts)

    # ---- lifecycle ----

    def stop(self):
        self._queue.join()
        self._queue.put(None)
        self._worker.join(timeout=5)
        os.killpg(os.getpgid(self.server.pid), signal.SIGKILL)
        self.server.wait()