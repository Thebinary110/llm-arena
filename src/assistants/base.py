"""Abstract base class and shared data structures for all assistant implementations."""

from __future__ import annotations

import asyncio
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, AsyncGenerator

from src.memory.conversation_memory import ConversationMemory

if TYPE_CHECKING:
    from src.tools.tool_registry import ToolRegistry
    from src.memory.structured_memory import StructuredMemoryManager

_TAG_RE = re.compile(r"\[REMEMBER:[^\]]*\]", re.IGNORECASE)
_SAFE_NAME_RE = re.compile(r"[^a-zA-Z0-9_]")


def _sanitize_collection_name(model_name: str, max_len: int = 40) -> str:
    """Strip all non-alphanumeric/underscore chars, ensure ChromaDB name is valid.

    ChromaDB requires names that are 3-512 chars, alphanumeric and underscores only
    (some versions also allow hyphens and dots, but using only [a-zA-Z0-9_] is safe
    across all versions and avoids the colon bug with names like 'qwen2.5:0.5b').
    """
    name = _SAFE_NAME_RE.sub("_", model_name)
    name = name.strip("_")
    name = name[:max_len].strip("_")
    if len(name) < 3:
        name = name + "_mm"
    return name


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
    """Abstract assistant defining the public interface: chat() and chat_stream_async().

    Subclasses implement _call_model() and optionally _stream_model() for their backend.
    """

    def __init__(
        self,
        model_name: str,
        system_prompt: str,
        config,
        tool_registry: "ToolRegistry | None" = None,
        user_id: str = "default",
        safety_filter=None,
        working_memory_size: int = 5,
    ) -> None:
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.config = config
        self.memory = ConversationMemory(config.CONVERSATION_MAX_TURNS)
        self._tool_registry = tool_registry
        self._user_id = user_id
        self._safety_filter = safety_filter
        self._structured_memory: "StructuredMemoryManager | None" = None

        if tool_registry:
            self.system_prompt += tool_registry.get_system_prompt_addition()

        self.system_prompt += (
            "\n\nYou can store important information using the tag [REMEMBER: <fact>]. "
            "Relevant past context will be injected automatically when available."
        )

        # Instantiate exactly once here in __init__. Never re-instantiated per turn.
        try:
            from src.memory.structured_memory import StructuredMemoryManager
            collection_prefix = _sanitize_collection_name(model_name)
            self._structured_memory = StructuredMemoryManager(
                user_id=user_id,
                collection_prefix=collection_prefix,
                working_memory_size=working_memory_size,
            )
        except Exception as exc:
            import logging
            logging.getLogger(__name__).warning("StructuredMemory unavailable: %s", exc)

    @abstractmethod
    def _call_model(self, messages: list[dict]) -> AssistantResponse:
        """Send messages to the backend and return a structured response.

        Implementations must catch all exceptions and return an AssistantResponse
        with the error field set rather than propagating exceptions.
        """

    async def _stream_model(self, messages: list[dict]) -> AsyncGenerator[str, None]:
        """Yield string chunks as the model generates output.

        Default implementation wraps _call_model as a single chunk.
        Subclasses override this for true token-by-token streaming.
        """
        response = await asyncio.to_thread(self._call_model, messages)
        if not response.is_error:
            yield response.content

    def _build_messages(self, memory_context: str = "") -> list[dict]:
        """Construct the full messages list: system prompt + memory context + history."""
        system = self.system_prompt
        if memory_context:
            system = f"{system}\n\n[Relevant memory context]:\n{memory_context}"
        messages: list[dict] = [{"role": "system", "content": system}]
        messages.extend(self.memory.get_history())
        return messages

    @staticmethod
    def _strip_memory_tags(text: str) -> str:
        return _TAG_RE.sub("", text).strip()

    def chat(self, user_input: str) -> AssistantResponse:
        """Public entry point: record input, call model, record output, return response.

        Preserved intact -- evaluation pipeline depends on this interface.
        """
        self.memory.add_user_message(user_input)
        context = ""
        if self._structured_memory:
            try:
                context = self._structured_memory.get_context(user_input)
            except Exception:
                pass
        messages = self._build_messages(memory_context=context)
        response = self._call_model(messages)
        if not response.is_error:
            clean = self._strip_memory_tags(response.content)
            self.memory.add_assistant_message(clean)
            if self._structured_memory:
                try:
                    self._structured_memory.add_turn(user_input, response.content)
                except Exception:
                    pass
        return response

    async def chat_stream_async(self, user_input: str) -> AsyncGenerator[str, None]:
        """Async streaming chat with tool loop and structured memory integration.

        Yields content string chunks as they arrive from the model.
        Tool calls are detected, executed, and the follow-up is streamed inline.
        """
        self.memory.add_user_message(user_input)
        context = ""
        if self._structured_memory:
            try:
                context = await asyncio.to_thread(
                    self._structured_memory.get_context, user_input
                )
            except Exception:
                pass

        messages = self._build_messages(memory_context=context)

        # First pass -- stream and collect full response
        full_content = ""
        async for chunk in self._stream_model(messages):
            full_content += chunk
            yield chunk

        # Tool loop -- at most one tool call per turn
        if self._tool_registry:
            tool_result = await asyncio.to_thread(
                self._tool_registry.detect_and_execute, full_content
            )
            if tool_result and tool_result.success:
                tool_block = f"\n\n[Search result]:\n{tool_result.output}\n"
                yield tool_block

                messages.append({"role": "assistant", "content": full_content})
                messages.append({
                    "role": "user",
                    "content": (
                        f"The web search for '{tool_result.query}' returned:\n"
                        f"{tool_result.output}\n\n"
                        "Now answer the original question using this information."
                    ),
                })
                followup = ""
                async for chunk in self._stream_model(messages):
                    followup += chunk
                    yield chunk
                full_content = full_content + tool_block + followup

        response_is_safe = True
        if self._safety_filter is not None:
            try:
                safety_result = await asyncio.to_thread(self._safety_filter.check, full_content)
                kw_hit = self._safety_filter.hard_filter(full_content)
                if safety_result.is_toxic or kw_hit is not None:
                    response_is_safe = False
                    logger.warning(
                        "OSS toxic response blocked before memory | score=%.3f | kw=%s",
                        safety_result.toxicity_score,
                        kw_hit,
                    )
                    if self._structured_memory:
                        self._structured_memory.clear_working()
                    self.memory.add_assistant_message(
                        "[OSS blocked: the model generated unsafe content that was not stored.]"
                    )
            except Exception as exc:
                logger.error("Pre-memory safety check error: %s", exc)

        if response_is_safe:
            clean = self._strip_memory_tags(full_content)
            self.memory.add_assistant_message(clean)
            if self._structured_memory:
                try:
                    await asyncio.to_thread(
                        self._structured_memory.add_turn, user_input, full_content
                    )
                except Exception:
                    pass

    def reset(self) -> None:
        """Clear conversation history to start a fresh session."""
        self.memory.clear()
