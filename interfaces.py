from abc import ABC, abstractmethod
from typing import Iterator


class InputInterface(ABC):
    @abstractmethod
    def get_input(self) -> str:
        """Block until input is received. Returns input as string."""
    
    @abstractmethod
    def has_input(self) -> bool:
        """Check if something has been inputted during generation. Used for mid-generation interruption."""


class OutputInterface(ABC):
    @abstractmethod
    def send(self, token: str) -> None:
        """Receive a single token for output. Module handles buffering internally if needed."""

    @abstractmethod
    def interrupt(self) -> None:
        """Stop current output immediately and clear any pending buffer."""
        

class CoreInterface(ABC):
    @abstractmethod
    def generate(self, user_input: str, context: list[dict], memories: str) -> Iterator[str]:
        """Takes user input, conversation context, and relevant memories. Yields response tokens."""

    @abstractmethod
    def set_system_prompt(self, prompt: str) -> None:
        """Set the system prompt used for all subsequent generations."""
    
    @abstractmethod
    def interrupt(self) -> None:
        """Interrupt LLM generation"""


class ShortTermMemoryInterface(ABC):
    @abstractmethod
    def add(self, role: str, content: str) -> None:
        """Add a message to the current session history."""

    @abstractmethod
    def get(self) -> list[dict]:
        """Return the full conversation history for the current session."""

    @abstractmethod
    def clear(self) -> None:
        """Clear the current session history."""


class LongTermMemoryInterface(ABC):
    @abstractmethod
    def store(self, messages: list[dict]) -> None:
        """Extract and persist important facts from a conversation exchange."""

    @abstractmethod
    def retrieve(self, query: str) -> str:
        """Retrieve relevant memories based on semantic similarity to the query."""