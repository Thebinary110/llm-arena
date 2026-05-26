"""Base evaluation infrastructure: EvalResult dataclass and BaseEvaluator ABC."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from src.assistants.base import AssistantResponse


@dataclass
class EvalResult:
    """Single evaluation result produced by any evaluator for any model.

    score: 0.0–1.0 where higher values always mean better model behaviour.
    label: coarse classification — 'pass', 'partial', or 'fail'.
    """

    prompt_id: str
    category: str           # "factual" | "adversarial" | "bias"
    model_name: str
    prompt: str
    response: str
    score: float            # 0.0 to 1.0
    label: str              # "pass" | "fail" | "partial"
    reasoning: str
    latency_ms: float
    is_toxic: bool = False
    toxicity_score: float = 0.0


class BaseEvaluator(ABC):
    """Abstract evaluator — all concrete evaluators implement exactly one method."""

    @abstractmethod
    def evaluate(self, prompt: dict, response: AssistantResponse) -> EvalResult:
        """Evaluate a single (prompt, response) pair and return an EvalResult.

        Args:
            prompt: dict with at minimum 'id', 'prompt', and evaluation-specific fields.
            response: the AssistantResponse to be judged.

        Returns:
            EvalResult with all fields populated.
        """
