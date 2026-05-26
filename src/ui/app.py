"""Gradio side-by-side chat UI for the dual AI assistant comparison system."""

from __future__ import annotations

import logging

import gradio as gr
from rich.logging import RichHandler

from src.assistants.base import BaseAssistant
from src.guardrails.safety_filter import SafetyFilter

logging.basicConfig(handlers=[RichHandler(rich_tracebacks=True)], level=logging.INFO)
logger = logging.getLogger(__name__)

# ── UI string constants ────────────────────────────────────────────────────────
TITLE = "Dual AI Assistant Comparison"
SUBTITLE = "OSS (Qwen 2.5-0.5B) vs Frontier (Llama 3.3-70B via Groq)"
OSS_LABEL = "OSS Assistant (Qwen 2.5-0.5B)"
FRONTIER_LABEL = "Frontier Assistant (Llama 3.3-70B)"
INPUT_PLACEHOLDER = "Type your message here and press Send..."
SEND_BUTTON_LABEL = "Send"
CLEAR_BUTTON_LABEL = "Clear Both"
SAFETY_ACCORDION_LABEL = "Safety Analysis (last response)"
USER_TOXIC_WARNING = "⚠️ Your message was flagged for potentially toxic content. Sending anyway."
RESPONSE_TOXIC_SUFFIX = "\n\n[SAFETY WARNING: This response was flagged for toxic content]"
ERROR_PREFIX = "❌ Error: "
OSS_LATENCY_LABEL = "OSS latency"
FRONTIER_LATENCY_LABEL = "Frontier latency"


def build_app(oss_assistant: BaseAssistant, frontier_assistant: BaseAssistant) -> gr.Blocks:
    """Construct and return the Gradio Blocks application.

    The UI knows nothing about which model is behind each assistant — it
    only calls the chat() method defined on BaseAssistant.
    """
    safety_filter = SafetyFilter(threshold=0.7)

    def send_message(
        user_input: str,
        oss_history: list[dict],
        frontier_history: list[dict],
    ) -> tuple[str, list[dict], list[dict], str]:
        """Process a user message: run safety check, call both models, update histories."""
        if not user_input.strip():
            return "", oss_history, frontier_history, _format_safety_md(None, None)

        # 1. Check user input toxicity (log but do not block)
        user_safety = safety_filter.check(user_input)
        if user_safety.is_toxic:
            logger.warning(
                "User input flagged as toxic (score=%.2f, categories=%s)",
                user_safety.toxicity_score,
                user_safety.flagged_categories,
            )

        # 2. Call both assistants
        oss_response = oss_assistant.chat(user_input)
        frontier_response = frontier_assistant.chat(user_input)

        # 3. Check response toxicity
        oss_safety = safety_filter.check(oss_response.content or oss_response.error or "")
        frontier_safety = safety_filter.check(
            frontier_response.content or frontier_response.error or ""
        )

        # 4. Build display content
        if oss_response.is_error:
            oss_display = f"{ERROR_PREFIX}{oss_response.error}"
        else:
            oss_display = oss_response.content
            if oss_safety.is_toxic:
                oss_display += RESPONSE_TOXIC_SUFFIX

        if frontier_response.is_error:
            frontier_display = f"{ERROR_PREFIX}{frontier_response.error}"
        else:
            frontier_display = frontier_response.content
            if frontier_safety.is_toxic:
                frontier_display += RESPONSE_TOXIC_SUFFIX

        # 5. Append to Gradio message histories (type="messages" format)
        oss_history = list(oss_history) + [
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": oss_display},
        ]
        frontier_history = list(frontier_history) + [
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": frontier_display},
        ]

        safety_md = _format_safety_md(oss_safety, frontier_safety, oss_response, frontier_response)

        return "", oss_history, frontier_history, safety_md

    def clear_both() -> tuple[list, list, str]:
        """Reset both assistants and return empty histories."""
        oss_assistant.reset()
        frontier_assistant.reset()
        return [], [], ""

    def _format_safety_md(oss_safety, frontier_safety, oss_resp=None, frontier_resp=None) -> str:
        """Format the safety analysis markdown string for the accordion panel."""
        if oss_safety is None and frontier_safety is None:
            return "_No messages yet._"

        lines = []

        if oss_safety is not None:
            oss_flag = "🔴 TOXIC" if oss_safety.is_toxic else "🟢 Clean"
            lines.append(f"**OSS Assistant:** {oss_flag}  ")
            lines.append(f"  - Toxicity score: `{oss_safety.toxicity_score:.3f}`  ")
            if oss_safety.flagged_categories:
                lines.append(f"  - Flagged: `{', '.join(oss_safety.flagged_categories)}`  ")
            if oss_resp:
                lines.append(f"  - Latency: `{oss_resp.latency_ms:.0f} ms`  ")
                if oss_resp.tokens_used:
                    lines.append(f"  - Tokens: `{oss_resp.tokens_used}`  ")

        lines.append("")

        if frontier_safety is not None:
            fr_flag = "🔴 TOXIC" if frontier_safety.is_toxic else "🟢 Clean"
            lines.append(f"**Frontier Assistant:** {fr_flag}  ")
            lines.append(f"  - Toxicity score: `{frontier_safety.toxicity_score:.3f}`  ")
            if frontier_safety.flagged_categories:
                lines.append(f"  - Flagged: `{', '.join(frontier_safety.flagged_categories)}`  ")
            if frontier_resp:
                lines.append(f"  - Latency: `{frontier_resp.latency_ms:.0f} ms`  ")
                if frontier_resp.tokens_used:
                    lines.append(f"  - Tokens: `{frontier_resp.tokens_used}`  ")

        return "\n".join(lines)

    # ── Layout ─────────────────────────────────────────────────────────────────
    with gr.Blocks(title=TITLE, theme=gr.themes.Soft()) as demo:
        gr.Markdown(f"# {TITLE}")
        gr.Markdown(f"*{SUBTITLE}*")

        oss_history_state = gr.State([])
        frontier_history_state = gr.State([])

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown(f"### {OSS_LABEL}")
                oss_chatbot = gr.Chatbot(
                    label=OSS_LABEL,
                    type="messages",
                    height=450,
                    show_copy_button=True,
                )
            with gr.Column(scale=1):
                gr.Markdown(f"### {FRONTIER_LABEL}")
                frontier_chatbot = gr.Chatbot(
                    label=FRONTIER_LABEL,
                    type="messages",
                    height=450,
                    show_copy_button=True,
                )

        with gr.Row():
            user_input = gr.Textbox(
                placeholder=INPUT_PLACEHOLDER,
                show_label=False,
                scale=8,
                lines=1,
                autofocus=True,
            )
            send_btn = gr.Button(SEND_BUTTON_LABEL, variant="primary", scale=1)
            clear_btn = gr.Button(CLEAR_BUTTON_LABEL, variant="secondary", scale=1)

        with gr.Accordion(SAFETY_ACCORDION_LABEL, open=False):
            safety_display = gr.Markdown("_No messages yet._")

        # ── event wiring ───────────────────────────────────────────────────────
        send_inputs = [user_input, oss_history_state, frontier_history_state]
        send_outputs = [user_input, oss_history_state, frontier_history_state, safety_display]

        send_btn.click(
            fn=send_message,
            inputs=send_inputs,
            outputs=send_outputs,
        ).then(
            fn=lambda h: h,
            inputs=[oss_history_state],
            outputs=[oss_chatbot],
        ).then(
            fn=lambda h: h,
            inputs=[frontier_history_state],
            outputs=[frontier_chatbot],
        )

        user_input.submit(
            fn=send_message,
            inputs=send_inputs,
            outputs=send_outputs,
        ).then(
            fn=lambda h: h,
            inputs=[oss_history_state],
            outputs=[oss_chatbot],
        ).then(
            fn=lambda h: h,
            inputs=[frontier_history_state],
            outputs=[frontier_chatbot],
        )

        clear_btn.click(
            fn=clear_both,
            outputs=[oss_history_state, frontier_history_state, safety_display],
        ).then(
            fn=lambda: [],
            outputs=[oss_chatbot],
        ).then(
            fn=lambda: [],
            outputs=[frontier_chatbot],
        )

    return demo
