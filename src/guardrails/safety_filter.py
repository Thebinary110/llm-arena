"""Real-time toxicity classifier using Detoxify.

Responsibility: classify text and return a SafetyResult.
This module never modifies or blocks text — that decision belongs to the caller.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from rich.logging import RichHandler

logging.basicConfig(handlers=[RichHandler(rich_tracebacks=True)], level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SafetyResult:
    """Toxicity classification result from Detoxify."""

    is_toxic: bool
    toxicity_score: float
    categories: dict[str, float] = field(default_factory=dict)
    flagged_categories: list[str] = field(default_factory=list)


class SafetyFilter:
    """Wraps Detoxify for lazy-loaded, threshold-based toxicity classification.

    The model is loaded on the first call to check(), not at import time,
    to avoid slowing application startup. If Detoxify fails to load,
    the filter degrades gracefully and logs a warning.
    """

    def __init__(self, threshold: float = 0.7) -> None:
        self.threshold = threshold
        self._model = None
        self._available: bool = True  # becomes False if loading ever fails

    def _ensure_loaded(self) -> None:
        """Load Detoxify model on first use."""
        if self._model is not None or not self._available:
            return
        try:
            from detoxify import Detoxify  # noqa: PLC0415

            self._model = Detoxify("original")
            logger.info("Detoxify model loaded successfully.")
        except Exception as exc:
            self._available = False
            logger.warning("Detoxify failed to load — guardrail disabled. Error: %s", exc)

    def check(self, text: str) -> SafetyResult:
        """Classify text and return a SafetyResult.

        Returns a safe (non-toxic) SafetyResult if the model is unavailable.
        """
        self._ensure_loaded()

        if not self._available or self._model is None:
            return SafetyResult(is_toxic=False, toxicity_score=0.0)

        try:
            scores: dict[str, float] = self._model.predict(text)
            # Normalise: Detoxify may return numpy floats
            scores = {k: float(v) for k, v in scores.items()}
            flagged = [k for k, v in scores.items() if v > self.threshold]
            toxicity_score = scores.get("toxicity", 0.0)
            is_toxic = len(flagged) > 0
            return SafetyResult(
                is_toxic=is_toxic,
                toxicity_score=toxicity_score,
                categories=scores,
                flagged_categories=flagged,
            )
        except Exception as exc:
            logger.error("Detoxify prediction failed: %s", exc)
            return SafetyResult(is_toxic=False, toxicity_score=0.0)
