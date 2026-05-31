import subprocess
import time
import requests
import threading
import queue
import os
from mem0 import Memory
from interfaces import LongTermMemoryInterface


class Module(LongTermMemoryInterface):
    def __init__(self, model_path, n_gpu_layers, **kwargs):
        self.user_id = "user"
        self.server = subprocess.Popen(
            ["python", "-m", "llama_cpp.server", "--model", model_path, "--port", "8080", "--n_gpu_layers", str(n_gpu_layers)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            preexec_fn=os.setsid
        )
        self._wait_for_server()
        self.memory = Memory.from_config({
            "llm": {
                "provider": "openai",
                "config": {
                    "model": "local-model",
                    "openai_base_url": "http://localhost:8080/v1",
                    "api_key": "dummy"
                }
            },
            "embedder": {
                "provider": "huggingface",
                "config": {
                    "model": "Qwen/Qwen3-Embedding-0.6B",
                    "model_kwargs": {"device": "cpu"}
                }
            },
            "vector_store": {
                "provider": "chroma",
                "config": {
                    "collection_name": "cybrex",
                    "path": "memory/chroma"
                }
            }
        })
        self._queue = queue.Queue()
        self._worker = threading.Thread(target=self._process_queue, daemon=True)
        self._worker.start()

    def _wait_for_server(self, timeout: int = 60):
        start = time.time()
        while time.time() - start < timeout:
            try:
                requests.get("http://localhost:8080/health")
                return
            except requests.exceptions.ConnectionError:
                time.sleep(1)
        raise TimeoutError("llama-server failed to start within timeout")

    def _process_queue(self):
        while True:
            messages = self._queue.get()
            if messages is None:
                self._queue.task_done()
                break
            try:
                self.memory.add(messages, user_id=self.user_id)
            except Exception as e:
                print(f"Memory store failed: {e}")
            self._queue.task_done()

    def stop(self):
        self._queue.join()
        self._queue.put(None)
        self.server.kill()
        self.server.wait()

    def store(self, messages: list[dict]) -> None:
        self._queue.put(messages)

    def retrieve(self, query: str) -> str:
        results = self.memory.search(query=query, user_id=self.user_id, limit=3)
        return "\n".join(f"- {entry['memory']}" for entry in results["results"])