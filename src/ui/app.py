"""Gradio side-by-side chat UI for the dual AI assistant comparison system."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import AsyncGenerator

import gradio as gr
from rich.logging import RichHandler

from src.assistants.base import BaseAssistant
from src.guardrails.safety_filter import SafetyFilter, SafetyResult

logging.basicConfig(handlers=[RichHandler(rich_tracebacks=True)], level=logging.INFO)
logger = logging.getLogger(__name__)

# -- UI string constants -------------------------------------------------------
TITLE = "[Arena] LLM Arena"
SUBTITLE = "OSS (Qwen 2.5-0.5B) vs Frontier (Llama 3.3-70B via Groq)"
OSS_LABEL = "OSS Assistant -- Qwen 2.5-0.5B"
FRONTIER_LABEL = "Frontier Assistant -- Llama 3.3-70B"
INPUT_PLACEHOLDER = "Type your message and press Enter or click Send..."
SEND_BUTTON_LABEL = "Send"
CLEAR_BUTTON_LABEL = "Clear Both"
SAFETY_ACCORDION_LABEL = "[Safety] Analysis (last response)"
RESPONSE_TOXIC_SUFFIX = "\n\n[WARNING] This response was flagged for toxic content."
ERROR_PREFIX = "[Error] "
CURSOR = "|"


def build_app(
    oss_assistant: BaseAssistant,
    frontier_assistant: BaseAssistant,
) -> gr.Blocks:
    """Construct and return the Gradio Blocks application."""
    # Bug 2 fix: use config threshold instead of hardcoded 0.7
    safety_filter = SafetyFilter(threshold=oss_assistant.config.TOXICITY_THRESHOLD)

    # -- helpers ---------------------------------------------------------------
    def _format_safety_md(
        oss_safety,
        frontier_safety,
        oss_latency_ms: float | None = None,
        frontier_latency_ms: float | None = None,
    ) -> str:
        if oss_safety is None and frontier_safety is None:
            return "_No messages yet._"
        lines: list[str] = []
        if oss_safety is not None:
            flag = "[TOXIC]" if oss_safety.is_toxic else "[Clean]"
            lines.append(f"**OSS Assistant:** {flag}")
            lines.append(f"- Toxicity score: `{oss_safety.toxicity_score:.3f}`")
            if oss_safety.flagged_categories:
                lines.append(f"- Flagged: `{', '.join(oss_safety.flagged_categories)}`")
            # Bug 3 fix: show latency and explicit token N/A for streaming
            if oss_latency_ms is not None:
                lines.append(f"- Latency: `{oss_latency_ms:.0f} ms`")
            lines.append("- Tokens: `N/A (streaming)`")
        lines.append("")
        if frontier_safety is not None:
            flag = "[TOXIC]" if frontier_safety.is_toxic else "[Clean]"
            lines.append(f"**Frontier Assistant:** {flag}")
            lines.append(f"- Toxicity score: `{frontier_safety.toxicity_score:.3f}`")
            if frontier_safety.flagged_categories:
                lines.append(f"- Flagged: `{', '.join(frontier_safety.flagged_categories)}`")
            if frontier_latency_ms is not None:
                lines.append(f"- Latency: `{frontier_latency_ms:.0f} ms`")
            lines.append("- Tokens: `N/A (streaming)`")
        return "\n".join(lines)

    def _format_memory_md(label: str, assistant: BaseAssistant) -> str:
        sm = getattr(assistant, "_structured_memory", None)
        if sm is None:
            return f"**{label}** -- _Memory not available._"
        try:
            data = sm.get_all_facts()
        except Exception as exc:
            return f"**{label}** -- _Error reading memory: {exc}_"

        lines = [f"### {label}"]
        if data.get("facts"):
            lines.append("**Stored facts:**")
            lines.extend(f"- {f}" for f in data["facts"])
            lines.append("")
        if data.get("episodes"):
            lines.append("**Recent turns (working memory):**")
            lines.extend(f"- {e}" for e in data["episodes"])
        if not data.get("facts") and not data.get("episodes"):
            lines.append("_No memories stored yet._")
        return "\n".join(lines)

    # -- streaming handler (parallel async) ------------------------------------
    async def send_message_stream(
        user_input: str,
        oss_history: list[dict],
        frontier_history: list[dict],
    ) -> AsyncGenerator:
        if not user_input.strip():
            yield (
                "",
                oss_history,
                frontier_history,
                oss_history,
                frontier_history,
                _format_safety_md(None, None),
            )
            return

        user_safety = safety_filter.check(user_input)
        if user_safety.is_toxic:
            logger.warning(
                "Input flagged toxic | score=%.2f | categories=%s",
                user_safety.toxicity_score,
                user_safety.flagged_categories,
            )

        # Append user message and empty assistant placeholder to both histories
        oss_history = list(oss_history) + [
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": CURSOR},
        ]
        frontier_history = list(frontier_history) + [
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": CURSOR},
        ]

        yield "", oss_history, frontier_history, oss_history, frontier_history, "_Generating..._"

        oss_queue: asyncio.Queue[str | None] = asyncio.Queue()
        frontier_queue: asyncio.Queue[str | None] = asyncio.Queue()

        async def _fill(queue: asyncio.Queue, assistant: BaseAssistant) -> None:
            try:
                async for chunk in assistant.chat_stream_async(user_input):
                    await queue.put(chunk)
            except Exception as exc:
                await queue.put(f"{ERROR_PREFIX}{exc}")
            await queue.put(None)

        # Bug 3 fix: track per-model stream latency from task start to completion
        stream_start = time.perf_counter()
        oss_task = asyncio.create_task(_fill(oss_queue, oss_assistant))
        frontier_task = asyncio.create_task(_fill(frontier_queue, frontier_assistant))

        oss_full = ""
        frontier_full = ""
        oss_done = False
        frontier_done = False
        oss_latency_ms: float | None = None
        frontier_latency_ms: float | None = None

        while not (oss_done and frontier_done):
            updated = False

            while not oss_queue.empty():
                chunk = oss_queue.get_nowait()
                if chunk is None:
                    oss_done = True
                    if oss_latency_ms is None:
                        oss_latency_ms = (time.perf_counter() - stream_start) * 1000
                else:
                    oss_full += chunk
                    updated = True

            while not frontier_queue.empty():
                chunk = frontier_queue.get_nowait()
                if chunk is None:
                    frontier_done = True
                    if frontier_latency_ms is None:
                        frontier_latency_ms = (time.perf_counter() - stream_start) * 1000
                else:
                    frontier_full += chunk
                    updated = True

            if updated:
                oss_display = oss_full + (CURSOR if not oss_done else "")
                frontier_display = frontier_full + (CURSOR if not frontier_done else "")
                oss_history[-1]["content"] = oss_display
                frontier_history[-1]["content"] = frontier_display
                yield (
                    "",
                    oss_history,
                    frontier_history,
                    oss_history,
                    frontier_history,
                    "_Generating..._",
                )

            if not (oss_done and frontier_done):
                await asyncio.sleep(0.04)

        await oss_task
        await frontier_task

        # Safety checks on completed responses
        # OSS: Detoxify + keyword filter
        oss_safety = safety_filter.check(oss_full)
        kw_hit = safety_filter.hard_filter(oss_full)
        if kw_hit is not None:
            merged_flags = list(set(oss_safety.flagged_categories) | {f"keyword:{kw_hit}"})
            oss_safety = SafetyResult(
                is_toxic=True,
                toxicity_score=max(oss_safety.toxicity_score, 1.0),
                categories=oss_safety.categories,
                flagged_categories=merged_flags,
            )

        # Frontier: Detoxify only
        frontier_safety = safety_filter.check(frontier_full)

        if oss_safety.is_toxic:
            oss_display = (
                "[OSS blocked] The OSS model attempted to generate unsafe content. "
                "This response was not stored in memory."
            )
        else:
            oss_display = oss_full

        frontier_display = frontier_full
        if frontier_safety.is_toxic:
            frontier_display += RESPONSE_TOXIC_SUFFIX

        oss_history[-1]["content"] = oss_display
        frontier_history[-1]["content"] = frontier_display

        safety_md = _format_safety_md(
            oss_safety, frontier_safety, oss_latency_ms, frontier_latency_ms
        )

        # Bug 3 fix: log safety_md length to verify non-empty before yield
        logger.info(
            "Turn complete | OSS %d chars | Frontier %d chars | safety_md %d chars",
            len(oss_full),
            len(frontier_full),
            len(safety_md),
        )

        yield (
            "",
            oss_history,
            frontier_history,
            oss_history,
            frontier_history,
            safety_md,
        )

    def clear_both() -> tuple:
        """Reset both assistant memories and wipe both chat histories."""
        oss_assistant.reset()
        frontier_assistant.reset()
        logger.info("Both assistants reset.")
        return [], [], "_No messages yet._", [], []

    def refresh_memory() -> tuple[str, str]:
        return (
            _format_memory_md(OSS_LABEL, oss_assistant),
            _format_memory_md(FRONTIER_LABEL, frontier_assistant),
        )

    def clear_oss_memory() -> str:
        sm = getattr(oss_assistant, "_structured_memory", None)
        if sm:
            sm.clear_all()
            oss_assistant.reset()
        return _format_memory_md(OSS_LABEL, oss_assistant)

    def clear_frontier_memory() -> str:
        sm = getattr(frontier_assistant, "_structured_memory", None)
        if sm:
            sm.clear_all()
            frontier_assistant.reset()
        return _format_memory_md(FRONTIER_LABEL, frontier_assistant)

    def clear_all_memory() -> tuple[str, str]:
        clear_oss_memory()
        clear_frontier_memory()
        return (
            _format_memory_md(OSS_LABEL, oss_assistant),
            _format_memory_md(FRONTIER_LABEL, frontier_assistant),
        )

    # -- layout ----------------------------------------------------------------
    with gr.Blocks(title=TITLE, theme=gr.themes.Soft()) as demo:

        gr.Markdown(f"# {TITLE}")
        gr.Markdown(f"*{SUBTITLE}*")

        oss_history_state      = gr.State([])
        frontier_history_state = gr.State([])

        with gr.Tabs():

            # Chat tab
            with gr.Tab("Chat"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown(f"### {OSS_LABEL}")
                        oss_chatbot = gr.Chatbot(
                            label=OSS_LABEL,
                            type="messages",
                            height=480,
                            show_copy_button=True,
                            avatar_images=(None, None),
                        )
                    with gr.Column(scale=1):
                        gr.Markdown(f"### {FRONTIER_LABEL}")
                        frontier_chatbot = gr.Chatbot(
                            label=FRONTIER_LABEL,
                            type="messages",
                            height=480,
                            show_copy_button=True,
                            avatar_images=(None, None),
                        )

                with gr.Row():
                    user_input = gr.Textbox(
                        placeholder=INPUT_PLACEHOLDER,
                        show_label=False,
                        scale=8,
                        lines=1,
                        autofocus=True,
                    )
                    send_btn  = gr.Button(SEND_BUTTON_LABEL,  variant="primary",   scale=1)
                    clear_btn = gr.Button(CLEAR_BUTTON_LABEL, variant="secondary", scale=1)

                # Bug 3 fix: open=True so metrics are visible without clicking
                with gr.Accordion(SAFETY_ACCORDION_LABEL, open=True):
                    safety_display = gr.Markdown("_No messages yet._")

            # Memory Inspector tab
            with gr.Tab("Memory Inspector"):
                with gr.Row():
                    refresh_btn        = gr.Button("[Refresh] Refresh",           variant="secondary")
                    clear_oss_btn      = gr.Button("[Clear] Clear OSS Memory",    variant="secondary")
                    clear_frontier_btn = gr.Button("[Clear] Clear Frontier Memory", variant="secondary")
                    clear_all_mem_btn  = gr.Button("[!] Clear All Memory",        variant="stop")

                with gr.Row():
                    with gr.Column(scale=1):
                        oss_memory_display = gr.Markdown("_Click Refresh to load._")
                    with gr.Column(scale=1):
                        frontier_memory_display = gr.Markdown("_Click Refresh to load._")

        # -- event wiring ------------------------------------------------------
        send_inputs  = [user_input, oss_history_state, frontier_history_state]
        send_outputs = [
            user_input,
            oss_history_state,
            frontier_history_state,
            oss_chatbot,
            frontier_chatbot,
            safety_display,
        ]

        def _chain(trigger_event):
            return trigger_event(
                fn=send_message_stream,
                inputs=send_inputs,
                outputs=send_outputs,
                trigger_mode="once",
            )

        _chain(send_btn.click)
        _chain(user_input.submit)

        clear_btn.click(
            fn=clear_both,
            outputs=[
                oss_history_state,
                frontier_history_state,
                safety_display,
                oss_chatbot,
                frontier_chatbot,
            ],
        )

        refresh_btn.click(
            fn=refresh_memory,
            outputs=[oss_memory_display, frontier_memory_display],
        )
        clear_oss_btn.click(
            fn=clear_oss_memory,
            outputs=[oss_memory_display],
        )
        clear_frontier_btn.click(
            fn=clear_frontier_memory,
            outputs=[frontier_memory_display],
        )
        clear_all_mem_btn.click(
            fn=clear_all_memory,
            outputs=[oss_memory_display, frontier_memory_display],
        )

    return demo
