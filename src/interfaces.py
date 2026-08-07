from abc import ABC, abstractmethod
from typing import Iterator


class InputInterface(ABC):
    @abstractmethod
    def get_input(self) -> str:
        pass

    @abstractmethod
    def has_input(self) -> bool:
        pass


class OutputInterface(ABC):
    @abstractmethod
    def send(self, token: str) -> None:
        pass

    @abstractmethod
    def interrupt(self) -> None:
        pass

    @abstractmethod
    def flush(self) -> None:
        pass

class CoreInterface(ABC):
    @abstractmethod
    def generate(self, user_input: str, context: list[dict], memories: str) -> Iterator[str]:
        pass

    @abstractmethod
    def set_system_prompt(self, prompt: str) -> None:
        pass

    @abstractmethod
    def interrupt(self) -> None:
        pass


class ContextInterface(ABC):
    @abstractmethod
    def add(self, role: str, content: str) -> None:
        pass

    @abstractmethod
    def get(self) -> list[dict]:
        pass

    @abstractmethod
    def clear(self) -> None:
        pass


class MemoryInterface(ABC):
    @abstractmethod
    def store(self, messages: list[dict]) -> None:
        pass

    @abstractmethod
    def retrieve(self, query: str) -> str:
        pass


class ControllerInterface(ABC):
    required_modules: list[str] = None

    def __init_subclass__(cls):
        super().__init_subclass__()
        if cls.required_modules is None:
            raise TypeError(f"{cls.__name__} must define required_modules")

    @abstractmethod
    def run(self, api) -> None:
        pass