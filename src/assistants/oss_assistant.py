"""OSS assistant backed by either a local Ollama server or a Modal cloud endpoint."""

from __future__ import annotations

import time

import requests
from rich.logging import RichHandler
import logging

from src.assistants.base import AssistantResponse, BaseAssistant

logging.basicConfig(handlers=[RichHandler(rich_tracebacks=True)], level=logging.INFO)
logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a helpful, harmless, and honest AI assistant. "
    "Answer clearly and concisely. "
    "If you are unsure about a fact, say so explicitly."
)


class OSSAssistant(BaseAssistant):
    """OSS assistant using Qwen2.5-0.5B-Instruct.

    Routes requests to a local Ollama server (development) or a Modal
    serverless endpoint (production) depending on config.USE_MODAL.
    """

    def __init__(self, config) -> None:
        super().__init__(
            model_name=config.OSS_MODEL_NAME,
            system_prompt=_SYSTEM_PROMPT,
            config=config,
        )

    def _call_model(self, messages: list[dict]) -> AssistantResponse:
        """Dispatch to Modal or Ollama based on runtime config."""
        if self.config.USE_MODAL:
            return self._call_modal(messages)
        return self._call_ollama(messages)

    def _call_modal(self, messages: list[dict]) -> AssistantResponse:
        """POST to the deployed Modal web endpoint."""
        url = self.config.MODAL_ENDPOINT.rstrip("/")
        payload = {"messages": messages, "max_tokens": 512, "temperature": 0.7}
        start = time.perf_counter()
        try:
            resp = requests.post(url, json=payload, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            latency_ms = (time.perf_counter() - start) * 1000
            return AssistantResponse(
                content=data["content"],
                model_name=self.model_name,
                latency_ms=latency_ms,
                tokens_used=data.get("tokens_used"),
            )
        except Exception as exc:
            latency_ms = (time.perf_counter() - start) * 1000
            logger.error("Modal call failed: %s", exc)
            return AssistantResponse(
                content="",
                model_name=self.model_name,
                latency_ms=latency_ms,
                error=str(exc),
            )

    def _call_ollama(self, messages: list[dict]) -> AssistantResponse:
        """POST to the local Ollama /api/chat endpoint."""
        url = self.config.OLLAMA_BASE_URL.rstrip("/") + "/api/chat"
        payload = {
            "model": self.config.OSS_MODEL_NAME,
            "messages": messages,
            "stream": False,
            "options": {"temperature": 0.7, "num_predict": 512},
        }
        start = time.perf_counter()
        try:
            resp = requests.post(url, json=payload, timeout=120)
            resp.raise_for_status()
            data = resp.json()
            latency_ms = (time.perf_counter() - start) * 1000
            content = data["message"]["content"]
            tokens_used = data.get("eval_count")
            return AssistantResponse(
                content=content,
                model_name=self.model_name,
                latency_ms=latency_ms,
                tokens_used=tokens_used,
            )
        except Exception as exc:
            latency_ms = (time.perf_counter() - start) * 1000
            logger.error("Ollama call failed: %s", exc)
            return AssistantResponse(
                content="",
                model_name=self.model_name,
                latency_ms=latency_ms,
                error=str(exc),
            )
