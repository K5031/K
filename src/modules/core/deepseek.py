import json
import os
import requests
from typing import Iterator
from interfaces import CoreInterface
from klogger import get_logger

DEEPSEEK_BASE_URL = "https://api.deepseek.com"


class Module(CoreInterface):
    """DeepSeek API-backed core. Same interface as the local llama.cpp core —
    controller.py doesn't need to know or care which one it's talking to."""

    def __init__(self, model="deepseek-v4-flash", **kwargs):
        self.log = get_logger("DeepSeek")
        self.model = model
        self.thinking = "disabled"  # hardcoded default — revisit once there's
        # a GUI to manage per-module settings without config schema drift
        self.system_prompt = "You are an AI assistant. Keep responses concise."
        self._interrupted = False

        self.api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError(
                "DeepSeek API key not found. Set the DEEPSEEK_API_KEY "
                "environment variable — never hardcode it in config or source."
            )

        self.log.info("Using model: %s (thinking=%s)", model, self.thinking)
        self._session = requests.Session()

    def interrupt(self):
        self._interrupted = True

    def set_system_prompt(self, prompt: str) -> None:
        self.system_prompt = prompt

    def generate(self, user_input: str, context: list[dict], memories: str) -> Iterator[str]:
        self._interrupted = False

        system_content = self.system_prompt
        if memories:
            system_content += f"\n\nRelevant memories about the user:\n{memories}"

        messages = [
            {"role": "system", "content": system_content},
            *context,
            {"role": "user", "content": user_input},
        ]

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "temperature": 0.45,
            "top_p": 0.85,
            "max_tokens": 512,
            "thinking": {"type": self.thinking},
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        resp = self._session.post(
            f"{DEEPSEEK_BASE_URL}/chat/completions",
            headers=headers,
            json=payload,
            stream=True,
            timeout=60,
        )

        try:
            resp.raise_for_status()
        except requests.HTTPError as e:
            self.log.error("DeepSeek API error: %s — %s", e, resp.text[:500])
            raise

        try:
            for line in resp.iter_lines():
                if self._interrupted:
                    break
                if not line:
                    continue
                line = line.decode("utf-8")
                if not line.startswith("data: "):
                    continue
                data = line[len("data: "):]
                if data.strip() == "[DONE]":
                    break
                chunk = json.loads(data)
                delta = chunk["choices"][0]["delta"]
                # reasoning_content holds thinking-mode output — kept separate
                # from the final answer so it never leaks into output/TTS
                token = delta.get("content", "")
                if token:
                    yield token
        finally:
            resp.close()