import json
import os
import requests
from typing import Iterator
from interfaces import CoreInterface
from klogger import get_logger

DEEPSEEK_BASE_URL = "https://api.deepseek.com"


class Module(CoreInterface):
    def __init__(self, model="deepseek-v4-flash", **kwargs):
        self.log = get_logger("DeepSeek")
        self.model = model
        self.thinking = "disabled"
        self.system_prompt = "You are an AI assistant. Keep responses concise."
        self._interrupted = False

        self.api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError(
                "DeepSeek API key not found. Set the DEEPSEEK_API_KEY "
                "environment variable."
            )

        self.log.info("Using model: %s (thinking=%s)", model, self.thinking)
        self._session = requests.Session()
        self.last_usage = None

    def interrupt(self):
        self._interrupted = True

    def set_system_prompt(self, prompt: str) -> None:
        self.system_prompt = prompt

    def generate(self, user_input: str, context: list[dict], memories: str) -> Iterator[str]:
        self._interrupted = False
        self.last_usage = None

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
            "stream_options": {"include_usage": True},
            "temperature": 0.45,
            "top_p": 0.85,
            "max_tokens": 512,
            "thinking": {"type": self.thinking},
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        try:
            resp = self._session.post(
                f"{DEEPSEEK_BASE_URL}/chat/completions",
                headers=headers,
                json=payload,
                stream=True,
                timeout=60,
            )
        except requests.exceptions.ConnectionError as e:
            self.log.error("DeepSeek connection failed: %s", e)
            yield "[connection error — check your network]"
            return
        except requests.exceptions.Timeout as e:
            self.log.error("DeepSeek request timed out: %s", e)
            yield "[request timed out]"
            return

        try:
            resp.raise_for_status()
        except requests.HTTPError as e:
            self.log.error("DeepSeek API error: %s — %s", e, resp.text[:500])
            yield "[API error — check logs]"
            return

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
                if chunk.get("usage"):
                    self.last_usage = chunk["usage"]
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                delta = choices[0]["delta"]
                token = delta.get("content", "")
                if token:
                    yield token
        except requests.exceptions.ConnectionError as e:
            self.log.error("DeepSeek connection dropped mid-response: %s", e)
            yield " [connection dropped]"
        finally:
            resp.close()