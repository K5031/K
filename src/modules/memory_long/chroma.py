import uuid
import os
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from interfaces import LongTermMemoryInterface
from config import BASE_DIR


class Module(LongTermMemoryInterface):
    def __init__(self, path: str = "data/chroma", collection_name: str = "k",
                 embedding_model: str = "Qwen/Qwen3-Embedding-0.6B", max_distance: float = 0.5, **kwargs):
        resolved_path = os.path.join(BASE_DIR, path)
        ef = SentenceTransformerEmbeddingFunction(model_name=embedding_model, device="cpu")
        client = chromadb.PersistentClient(path=resolved_path)
        self._collection = client.get_or_create_collection(
            name=collection_name,
            embedding_function=ef,
            metadata={"hnsw:space": "cosine"}
        )
        self._max_distance = max_distance

    def store(self, messages: list[dict]) -> None:
        text = "\n".join(f"{m['role']}: {m['content']}" for m in messages if m.get("content"))
        if not text.strip():
            return
        self._collection.add(documents=[text], ids=[str(uuid.uuid4())])

    def retrieve(self, query: str) -> str:
        count = self._collection.count()
        if count == 0:
            return ""
        results = self._collection.query(query_texts=[query], n_results=min(3, count))
        docs = results.get("documents", [[]])[0]
        distances = results.get("distances", [[]])[0]

        print(f"[debug] docs: {docs}")
        print(f"[debug] distances: {distances}")

        filtered = [doc for doc, dist in zip(docs, distances) if dist <= self._max_distance]
        if not filtered:
            return ""
        return "\n".join(f"- {doc}" for doc in filtered)
