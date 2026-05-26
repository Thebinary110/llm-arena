"""Manages sliding-window conversation history for an assistant session."""

from collections import deque
from dataclasses import dataclass


@dataclass
class Message:
    """Represents a single turn in the conversation."""

    role: str   # "user" | "assistant" | "system"
    content: str


class ConversationMemory:
    """Maintains a bounded deque of Messages with a max-turns sliding window.

    Responsibility: store and retrieve conversation history only.
    Does not know about models, APIs, or evaluation.
    """

    def __init__(self, max_turns: int) -> None:
        # Each turn = 1 user + 1 assistant message → maxlen = max_turns * 2
        self._history: deque[Message] = deque(maxlen=max_turns * 2)
        self._max_turns = max_turns

    def add_user_message(self, content: str) -> None:
        """Append a user message to the history."""
        self._history.append(Message(role="user", content=content))

    def add_assistant_message(self, content: str) -> None:
        """Append an assistant message to the history."""
        self._history.append(Message(role="assistant", content=content))

    def get_history(self) -> list[dict]:
        """Return history as a list of role/content dicts suitable for LLM APIs."""
        return [{"role": msg.role, "content": msg.content} for msg in self._history]

    def clear(self) -> None:
        """Remove all messages from history."""
        self._history.clear()

    def __len__(self) -> int:
        """Return total number of individual messages stored."""
        return len(self._history)


    @property
    def turn_count(self) -> int:
        """Return number of complete user-assistant turns."""
        return len(self._history) // 2
