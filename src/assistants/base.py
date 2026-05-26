"""Abstract base class and shared data structures for all assistant implementations."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from src.memory.conversation_memory import ConversationMemory


@dataclass
class AssistantResponse:
    """Canonical response object returned by every assistant implementation.

    The UI and evaluation engine interact exclusively with this type,
    ensuring zero coupling to specific model backends.
    """

    content: str
    model_name: str
    latency_ms: float
    tokens_used: int | None = None
    error: str | None = None

    @property
    def is_error(self) -> bool:
        """True when the response represents a failed API call."""
        return self.error is not None


class BaseAssistant(ABC):
    """Abstract assistant defining the single public interface: chat().

    Subclasses implement _call_model() for their specific backend.
    This class owns memory management and message construction so that
    neither the UI nor the evaluators need to know about conversation state.
    """

    def __init__(self, model_name: str, system_prompt: str, config) -> None:
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.config = config
        self.memory = ConversationMemory(config.CONVERSATION_MAX_TURNS)

    @abstractmethod
    def _call_model(self, messages: list[dict]) -> AssistantResponse:
        """Send messages to the backend and return a structured response.

        Implementations must catch all exceptions and return an AssistantResponse
        with the error field set rather than propagating exceptions.
        """

    def chat(self, user_input: str) -> AssistantResponse:
        """Public entry point: record input, call model, record output, return response."""
        self.memory.add_user_message(user_input)
        messages = self._build_messages()
        response = self._call_model(messages)
        if not response.is_error:
            self.memory.add_assistant_message(response.content)
        return response

    def reset(self) -> None:
        """Clear conversation history to start a fresh session."""
        self.memory.clear()

    def _build_messages(self) -> list[dict]:
        """Construct the full messages list: system prompt + conversation history."""
        messages: list[dict] = [{"role": "system", "content": self.system_prompt}]
        messages.extend(self.memory.get_history())
        return messages
