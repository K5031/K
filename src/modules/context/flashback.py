from llama_cpp import Llama

from interfaces import ContextInterface
from klogger import get_logger


class Module(ContextInterface):
    """Tracks conversation context for the current session, trimmed by
    exact token count (using the real model's tokenizer, loaded vocab-only —
    no weights, so it's fast and cheap) so it stays proportional to what
    actually fills core's context window."""

    def __init__(self, model_path: str, max_context_tokens: int = 6000):
        self.conversation = []
        self.max_context_tokens = max_context_tokens
        self.log = get_logger("Context")

        # vocab_only=True loads just the tokenizer, not the model weights —
        # fast, cheap, and gives exact counts matching core's real tokenizer.
        self.tokenizer = Llama(model_path=model_path, vocab_only=True, verbose=False)

    def _count_tokens(self, text: str) -> int:
        return len(self.tokenizer.tokenize(text.encode("utf-8")))

    def _total_tokens(self) -> int:
        return sum(self._count_tokens(m["content"]) for m in self.conversation)

    def add(self, role: str, content: str) -> None:
        self.conversation.append({"role": role, "content": content})

        dropped = 0
        while self._total_tokens() > self.max_context_tokens and len(self.conversation) > 1:
            self.conversation.pop(0)
            dropped += 1

        if dropped:
            self.log.info("trimmed %d oldest message(s) to stay under %d token budget "
                          "(now %d tokens, %d messages)",
                          dropped, self.max_context_tokens, self._total_tokens(),
                          len(self.conversation))

    def get(self) -> list[dict]:
        return self.conversation

    def clear(self) -> None:
        self.log.info("conversation cleared (%d messages, %d tokens)",
                      len(self.conversation), self._total_tokens())
        self.conversation = []