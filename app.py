"""HuggingFace Spaces entry point.

This file is exclusively for HF Spaces deployment.
For local development use: python main.py --mode chat
"""

from __future__ import annotations

import logging

from rich.logging import RichHandler

logging.basicConfig(
    level=logging.INFO,
    handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
    format="%(message)s",
    datefmt="[%X]",
)
logger = logging.getLogger(__name__)

try:
    from config import config
    from src.assistants.oss_assistant import OSSAssistant
    from src.assistants.frontier_assistant import FrontierAssistant
    from src.tools.tool_registry import ToolRegistry
    from src.tools.web_search import WebSearchTool
    from src.ui.app import build_app

    logger.info("Initialising tool registry and assistants...")
    registry = ToolRegistry()
    registry.register(WebSearchTool(max_results=3))

    oss = OSSAssistant(config=config, tool_registry=registry, user_id="oss")
    frontier = FrontierAssistant(config=config, tool_registry=registry, user_id="frontier")

    logger.info("Building Gradio app...")
    demo = build_app(oss, frontier)

    logger.info("Launching on 0.0.0.0:7860")
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)

except Exception as exc:
    logger.error("Failed to start app: %s", exc, exc_info=True)
    raise
