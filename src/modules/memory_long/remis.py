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

Examples:
Input: hi
Output: {"facts": []}
Input: my name is Alex and I work as a nurse
Output: {"facts": ["User's name is Alex", "User works as a nurse"]}
Input: what's your name?
Output: {"facts": []}

Conversation:
{conversation}

Return only valid JSON: {"facts": [...]}"""


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
    def __init__(self, model_path, n_gpu_layers, n_ctx=4096,
                 collection_name="k_memory",
                 embed_model_name="Qwen/Qwen3-Embedding-0.6B", **kwargs):
        self.user_id = "user"
        self.model_path = model_path
        print(f"[memory_long] Extraction model: {model_path}")

        self.server = subprocess.Popen(
            ["python", "-m", "llama_cpp.server",
             "--model", model_path,
             "--port", "8080",
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
        self.collection = self.client.get_or_create_collection(collection_name)
        self.embedder = SentenceTransformer(embed_model_name, device="cpu")

        self._queue = queue.Queue()
        self._worker = threading.Thread(target=self._process_queue, daemon=True)
        self._worker.start()

    def _wait_for_server(self, timeout: int = 60):
        start = time.time()
        while time.time() - start < timeout:
            try:
                r = requests.get("http://localhost:8080/v1/models")
                if r.status_code == 200:
                    return
            except requests.exceptions.ConnectionError:
                pass
            time.sleep(1)
        raise TimeoutError("llama-server failed to start within timeout")

    def _embed(self, text: str) -> list[float]:
        return self.embedder.encode(text).tolist()

    def _process_queue(self):
        while True:
            messages = self._queue.get()
            if messages is None:
                self._queue.task_done()
                break
            try:
                facts = self._extract_and_save(messages)
                if facts:
                    print(f"[memory_long] stored: {facts}")
            except Exception as e:
                print(f"Memory store failed: {e}")
            self._queue.task_done()

    def _extract_and_save(self, messages: list[dict]) -> list[str]:
        conversation = _format_conversation(messages)
        if not conversation.strip():
            return []

        prompt = EXTRACTION_PROMPT.replace("{conversation}", conversation)
        resp = requests.post(
            "http://localhost:8080/v1/chat/completions",
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

        ids = [str(uuid.uuid4()) for _ in facts]
        embeddings = [self._embed(f) for f in facts]
        now = datetime.now(timezone.utc).isoformat()
        metadatas = [{"data": f, "user_id": self.user_id, "created_at": now} for f in facts]
        self.collection.add(ids=ids, embeddings=embeddings, metadatas=metadatas)
        return facts

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

    # ---- lifecycle (not part of the interface, but Registry may call it) ----

    def stop(self):
        self._queue.join()
        self._queue.put(None)
        self._worker.join(timeout=5)
        os.killpg(os.getpgid(self.server.pid), signal.SIGKILL)
        self.server.wait()