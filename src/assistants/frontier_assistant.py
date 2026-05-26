"""Frontier assistant backed by Groq's hosted Llama-3.3-70B-Versatile."""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from typing import AsyncGenerator

from groq import Groq
from rich.logging import RichHandler

from src.assistants.base import AssistantResponse, BaseAssistant

logging.basicConfig(handlers=[RichHandler(rich_tracebacks=True)], level=logging.INFO)
logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a helpful, harmless, and honest AI assistant. "
    "Answer clearly and concisely. "
    "If you are unsure about a fact, say so explicitly."
)


class FrontierAssistant(BaseAssistant):
    """Frontier assistant using Llama-3.3-70B-Versatile via the Groq API.

    Uses the same system prompt as OSSAssistant for a fair side-by-side comparison.
    """

    def __init__(self, config, tool_registry=None, user_id: str = "default") -> None:
        super().__init__(
            model_name=config.FRONTIER_MODEL_NAME,
            system_prompt=_SYSTEM_PROMPT,
            config=config,
            tool_registry=tool_registry,
            user_id=user_id,
        )
        self.client = Groq(api_key=config.GROQ_API_KEY)

    def _call_model(self, messages: list[dict]) -> AssistantResponse:
        """Call the Groq chat completions endpoint and return a structured response."""
        start = time.perf_counter()
        try:
            completion = self.client.chat.completions.create(
                model=self.config.FRONTIER_MODEL_NAME,
                messages=messages,
                max_tokens=1024,
                temperature=0.7,
            )
            latency_ms = (time.perf_counter() - start) * 1000
            content = completion.choices[0].message.content
            tokens_used = completion.usage.total_tokens if completion.usage else None
            return AssistantResponse(
                content=content,
                model_name=self.model_name,
                latency_ms=latency_ms,
                tokens_used=tokens_used,
            )
        except Exception as exc:
            latency_ms = (time.perf_counter() - start) * 1000
            logger.error("Groq API call failed: %s", exc)
            return AssistantResponse(
                content="",
                model_name=self.model_name,
                latency_ms=latency_ms,
                error=str(exc),
            )

    async def _stream_model(self, messages: list[dict]) -> AsyncGenerator[str, None]:
        """Stream tokens from Groq using thread + asyncio queue."""
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[str | None] = asyncio.Queue()

        def _worker() -> None:
            try:
                stream = self.client.chat.completions.create(
                    model=self.config.FRONTIER_MODEL_NAME,
                    messages=messages,
                    max_tokens=1024,
                    temperature=0.7,
                    stream=True,
                )
                for chunk in stream:
                    content = chunk.choices[0].delta.content
                    if content:
                        loop.call_soon_threadsafe(queue.put_nowait, content)
            except Exception as exc:
                logger.error("Groq stream error: %s", exc)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)

        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()

        while True:
            chunk = await queue.get()
            if chunk is None:
                break
            yield chunk
