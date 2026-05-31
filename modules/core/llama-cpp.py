from typing import Iterator
from llama_cpp import Llama
from interfaces import CoreInterface


class Module(CoreInterface):
    def __init__(
        self,
        model_path,
        n_gpu_layers,
        n_ctx,
        **kwargs
    ):
        self.system_prompt = "You are an AI assistant. Keep responses concise."
        self.model_path = model_path
        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            offload_kqv=True,
            n_gpu_layers=n_gpu_layers,
            main_gpu=0,
            n_threads=4,
            n_threads_batch=4,
            n_batch=512,
            n_ubatch=512,
            use_mmap=True,
            cache=True,
            verbose=False,
        )
        self._interrupted = False
    
    def interrupt(self):
        self._interrupted = True    

    def set_system_prompt(self, prompt: str) -> None:
        self.system_prompt = prompt

    def generate(self, user_input: str, context: list[dict], memories: str) -> Iterator[str]:
        self._interrupted = False
        for chunk in self.llm.create_chat_completion(
            messages=[
                {"role": "system", "content": self.system_prompt},  # static — will be cached after first call
                *context,
                {"role": "user", "content": f"Relevant context:\n{memories}\n\n{user_input}" if memories else user_input},
            ],
            stream=True,
            temperature=0.45,
            top_p=0.85,
            repeat_penalty=1.1,
            max_tokens=512,
        ):
            if self._interrupted:
                break
            token = chunk["choices"][0]["delta"].get("content", "")
            yield token