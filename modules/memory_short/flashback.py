from interfaces import ShortTermMemoryInterface


class Module(ShortTermMemoryInterface):
    def __init__(self):
        self.conversation = []

    def add(self, role: str, content: str) -> None:
        self.conversation.append({"role": role, "content": content})

    def get(self) -> list[dict]:
        return self.conversation

    def clear(self) -> None:
        self.conversation = []