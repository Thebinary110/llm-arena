"""Abstract base class for all assistant tools."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class ToolResult:
    tool_name: str
    query: str
    output: str
    success: bool
    error: str | None = None


class BaseTool(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def description(self) -> str: ...

    @property
    @abstractmethod
    def usage_pattern(self) -> str: ...

    @abstractmethod
    def execute(self, query: str) -> ToolResult: ...
