"""Registry that maps model outputs to tool executions."""

from __future__ import annotations

import logging
import re

from rich.logging import RichHandler

from src.tools.base_tool import BaseTool, ToolResult

logging.basicConfig(handlers=[RichHandler(rich_tracebacks=True)], level=logging.INFO)
logger = logging.getLogger(__name__)

_SEARCH_RE = re.compile(r"SEARCH:\s*(.+)", re.IGNORECASE)


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        self._tools[tool.name] = tool
        logger.info("Registered tool: %s", tool.name)

    def get_system_prompt_addition(self) -> str:
        if not self._tools:
            return ""
        lines = [
            "\nYou have access to the following tools:",
        ]
        for tool in self._tools.values():
            lines.append(f"  - {tool.name}: {tool.description}")
            lines.append(f"    Usage: output exactly '{tool.usage_pattern}' on its own line.")
        lines.append(
            "Use a tool when you need current information or facts you are unsure about. "
            "Output the tool usage line, then stop — the result will be injected automatically."
        )
        return "\n".join(lines)

    def detect_and_execute(self, text: str) -> ToolResult | None:
        """Return a ToolResult if a tool call is detected in text, else None."""
        match = _SEARCH_RE.search(text)
        if match:
            query = match.group(1).strip()
            tool = self._tools.get("web_search")
            if tool:
                logger.info("Tool call detected | web_search | query=%r", query)
                return tool.execute(query)
        return None
