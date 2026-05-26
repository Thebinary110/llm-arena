"""DuckDuckGo web search tool."""

from __future__ import annotations

import logging

from rich.logging import RichHandler

from src.tools.base_tool import BaseTool, ToolResult

logging.basicConfig(handlers=[RichHandler(rich_tracebacks=True)], level=logging.INFO)
logger = logging.getLogger(__name__)


class WebSearchTool(BaseTool):
    """Searches the web via DuckDuckGo -- no API key required."""

    def __init__(self, max_results: int = 3) -> None:
        self.max_results = max_results

    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return "Search the web for current information using DuckDuckGo"

    @property
    def usage_pattern(self) -> str:
        return "SEARCH: <your search query>"

    def execute(self, query: str) -> ToolResult:
        try:
            from duckduckgo_search import DDGS

            results = list(DDGS().text(query, max_results=self.max_results))
            if not results:
                return ToolResult(
                    tool_name=self.name,
                    query=query,
                    output="No results found.",
                    success=True,
                )
            lines: list[str] = []
            for i, r in enumerate(results, 1):
                lines.append(f"[{i}] {r.get('title', '')}\n{r.get('href', '')}\n{r.get('body', '')}")
            output = "\n\n".join(lines)
            logger.info("WebSearchTool | query=%r | results=%d", query, len(results))
            return ToolResult(tool_name=self.name, query=query, output=output, success=True)
        except Exception as exc:
            logger.error("WebSearchTool failed: %s", exc)
            return ToolResult(
                tool_name=self.name,
                query=query,
                output="",
                success=False,
                error=str(exc),
            )
