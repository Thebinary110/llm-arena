"""Gradio side-by-side chat UI for the dual AI assistant comparison system."""

from __future__ import annotations

import asyncio
import logging
from typing import AsyncGenerator

import gradio as gr
from rich.logging import RichHandler

from src.assistants.base import BaseAssistant
from src.guardrails.safety_filter import SafetyFilter

logging.basicConfig(handlers=[RichHandler(rich_tracebacks=True)], level=logging.INFO)
logger = logging.getLogger(__name__)

# ── UI string constants ────────────────────────────────────────────────────────
TITLE = "⚔️ LLM Arena"
SUBTITLE = "OSS (Qwen 2.5-0.5B) vs Frontier (Llama 3.3-70B via Groq)"
OSS_LABEL = "OSS Assistant — Qwen 2.5-0.5B"
FRONTIER_LABEL = "Frontier Assistant — Llama 3.3-70B"
INPUT_PLACEHOLDER = "Type your message and press Enter or click Send..."
SEND_BUTTON_LABEL = "Send"
CLEAR_BUTTON_LABEL = "Clear Both"
SAFETY_ACCORDION_LABEL = "🛡️ Safety Analysis (last response)"
RESPONSE_TOXIC_SUFFIX = "\n\n⚠️ [SAFETY WARNING: This response was flagged for toxic content]"
ERROR_PREFIX = "❌ Error: "
CURSOR = "▋"


def build_app(
    oss_assistant: BaseAssistant,
    frontier_assistant: BaseAssistant,
) -> gr.Blocks:
    """Construct and return the Gradio Blocks application."""
    safety_filter = SafetyFilter(threshold=0.7)

    # ── helpers ───────────────────────────────────────────────────────────────
    def _format_safety_md(oss_safety, frontier_safety) -> str:
        if oss_safety is None and frontier_safety is None:
            return "_No messages yet._"
        lines: list[str] = []
        if oss_safety is not None:
            flag = "🔴 TOXIC" if oss_safety.is_toxic else "🟢 Clean"
            lines.append(f"**OSS Assistant:** {flag}")
            lines.append(f"- Toxicity score: `{oss_safety.toxicity_score:.3f}`")
            if oss_safety.flagged_categories:
                lines.append(f"- Flagged: `{', '.join(oss_safety.flagged_categories)}`")
        lines.append("")
        if frontier_safety is not None:
            flag = "🔴 TOXIC" if frontier_safety.is_toxic else "🟢 Clean"
            lines.append(f"**Frontier Assistant:** {flag}")
            lines.append(f"- Toxicity score: `{frontier_safety.toxicity_score:.3f}`")
            if frontier_safety.flagged_categories:
                lines.append(f"- Flagged: `{', '.join(frontier_safety.flagged_categories)}`")
        return "\n".join(lines)

    def _format_memory_md(label: str, assistant: BaseAssistant) -> str:
        sm = getattr(assistant, "_structured_memory", None)
        if sm is None:
            return f"**{label}** — _Memory not available._"
        try:
            data = sm.get_all_facts()
        except Exception as exc:
            return f"**{label}** — _Error reading memory: {exc}_"

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

    # ── streaming handler (Feature 1 + 2) ────────────────────────────────────
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

        # Queues for parallel streaming (Feature 2)
        oss_queue: asyncio.Queue[str | None] = asyncio.Queue()
        frontier_queue: asyncio.Queue[str | None] = asyncio.Queue()

        async def _fill(queue: asyncio.Queue, assistant: BaseAssistant) -> None:
            try:
                async for chunk in assistant.chat_stream_async(user_input):
                    await queue.put(chunk)
            except Exception as exc:
                await queue.put(f"{ERROR_PREFIX}{exc}")
            await queue.put(None)

        oss_task = asyncio.create_task(_fill(oss_queue, oss_assistant))
        frontier_task = asyncio.create_task(_fill(frontier_queue, frontier_assistant))

        oss_full = ""
        frontier_full = ""
        oss_done = False
        frontier_done = False

        while not (oss_done and frontier_done):
            updated = False

            while not oss_queue.empty():
                chunk = oss_queue.get_nowait()
                if chunk is None:
                    oss_done = True
                else:
                    oss_full += chunk
                    updated = True

            while not frontier_queue.empty():
                chunk = frontier_queue.get_nowait()
                if chunk is None:
                    frontier_done = True
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
        oss_safety = safety_filter.check(oss_full)
        frontier_safety = safety_filter.check(frontier_full)

        oss_display = oss_full
        if oss_safety.is_toxic:
            oss_display += RESPONSE_TOXIC_SUFFIX
        frontier_display = frontier_full
        if frontier_safety.is_toxic:
            frontier_display += RESPONSE_TOXIC_SUFFIX

        oss_history[-1]["content"] = oss_display
        frontier_history[-1]["content"] = frontier_display

        safety_md = _format_safety_md(oss_safety, frontier_safety)

        logger.info(
            "Turn complete | OSS %d chars | Frontier %d chars",
            len(oss_full),
            len(frontier_full),
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

    # ── layout ────────────────────────────────────────────────────────────────
    with gr.Blocks(title=TITLE, theme=gr.themes.Soft()) as demo:

        gr.Markdown(f"# {TITLE}")
        gr.Markdown(f"*{SUBTITLE}*")

        oss_history_state      = gr.State([])
        frontier_history_state = gr.State([])

        with gr.Tabs():

            # ── Chat tab ──────────────────────────────────────────────────────
            with gr.Tab("💬 Chat"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown(f"### {OSS_LABEL}")
                        oss_chatbot = gr.Chatbot(
                            label=OSS_LABEL,
                            type="messages",
                            height=480,
                            show_copy_button=True,
                            avatar_images=(None, "🤖"),
                        )
                    with gr.Column(scale=1):
                        gr.Markdown(f"### {FRONTIER_LABEL}")
                        frontier_chatbot = gr.Chatbot(
                            label=FRONTIER_LABEL,
                            type="messages",
                            height=480,
                            show_copy_button=True,
                            avatar_images=(None, "🚀"),
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

                with gr.Accordion(SAFETY_ACCORDION_LABEL, open=False):
                    safety_display = gr.Markdown("_No messages yet._")

            # ── Memory Inspector tab (Feature 4) ──────────────────────────────
            with gr.Tab("🧠 Memory Inspector"):
                with gr.Row():
                    refresh_btn         = gr.Button("🔄 Refresh",           variant="secondary")
                    clear_oss_btn       = gr.Button("🗑 Clear OSS Memory",   variant="secondary")
                    clear_frontier_btn  = gr.Button("🗑 Clear Frontier Memory", variant="secondary")
                    clear_all_mem_btn   = gr.Button("⚠️ Clear All Memory",  variant="stop")

                with gr.Row():
                    with gr.Column(scale=1):
                        oss_memory_display = gr.Markdown("_Click Refresh to load._")
                    with gr.Column(scale=1):
                        frontier_memory_display = gr.Markdown("_Click Refresh to load._")

        # ── event wiring ──────────────────────────────────────────────────────
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

        # Memory Inspector wiring
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
