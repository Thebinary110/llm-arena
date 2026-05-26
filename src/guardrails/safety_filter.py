"""Real-time toxicity classifier using Detoxify with keyword pre-filter.

Responsibility: classify text and return a SafetyResult.
This module never modifies or blocks text -- that decision belongs to the caller.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from rich.logging import RichHandler

logging.basicConfig(handlers=[RichHandler(rich_tracebacks=True)], level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Keyword filter patterns -- compiled once at module load.
# Categories catch context-dependent harmful content that Detoxify misses
# (e.g. drug synthesis instructions score near 0.002 on Detoxify "original").
# ---------------------------------------------------------------------------

_KEYWORD_CATEGORIES: dict[str, list[str]] = {
    "drug_synthesis": [
        "methamphetamine",
        "meth synthesis",
        "drug manufacturing",
        "illicit drug synthesis",
        "illicit drug",
        "ephedrine",
        "pseudoephedrine",
        "precursor chemical",
        "reductive amination",
        "synthesize",
        "precursor",
        "reduction",
    ],
    "weapons": [
        "bomb making",
        "how to make a bomb",
        "detonator",
        "improvised explosive device",
        "ied construction",
        "weapon synthesis",
        "ammunition manufacturing",
        "explosive",
        "ied",
    ],
    "hacking": [
        "sql injection tutorial",
        "exploit code",
        "malware creation",
        "ransomware deployment",
        "malware",
        "keylogger",
        "ransomware",
    ],
}


def _compile_category(phrases: list[str]) -> list[re.Pattern]:
    patterns: list[re.Pattern] = []
    for phrase in phrases:
        escaped = re.escape(phrase)
        if " " in phrase:
            # Multi-word phrase: match verbatim, case-insensitive
            patterns.append(re.compile(escaped, re.IGNORECASE))
        else:
            # Single word: require word boundaries to avoid partial matches
            patterns.append(re.compile(r"\b" + escaped + r"\b", re.IGNORECASE))
    return patterns


_COMPILED_KEYWORDS: dict[str, list[re.Pattern]] = {
    cat: _compile_category(phrases)
    for cat, phrases in _KEYWORD_CATEGORIES.items()
}


@dataclass
class SafetyResult:
    """Toxicity classification result from Detoxify and/or keyword filter."""

    is_toxic: bool
    toxicity_score: float
    categories: dict[str, float] = field(default_factory=dict)
    flagged_categories: list[str] = field(default_factory=list)


class SafetyFilter:
    """Wraps Detoxify for lazy-loaded, threshold-based toxicity classification.

    The model is loaded on the first call to check(), not at import time,
    to avoid slowing application startup. If Detoxify fails to load,
    the filter degrades gracefully and logs a warning.

    hard_filter() provides a fast keyword-based secondary check for categories
    that Detoxify systematically under-scores (drug synthesis, weapon instructions,
    malicious code). Call it explicitly on OSS output; do not apply to frontier.
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
            logger.warning("Detoxify failed to load -- guardrail disabled. Error: %s", exc)

    def hard_filter(self, text: str) -> str | None:
        """Keyword-based filter for content Detoxify under-scores.

        Returns the matched category name if a keyword is found, else None.
        Intended for use on OSS output only -- call explicitly, not via check().
        """
        for category, patterns in _COMPILED_KEYWORDS.items():
            for pattern in patterns:
                if pattern.search(text):
                    logger.warning(
                        "Keyword filter triggered | category=%s | pattern=%s",
                        category,
                        pattern.pattern,
                    )
                    return category
        return None

    def check(self, text: str) -> SafetyResult:
        """Classify text via Detoxify and return a SafetyResult.

        Returns a safe (non-toxic) SafetyResult if the model is unavailable.
        Does NOT call hard_filter() -- the caller is responsible for combining
        results when keyword filtering is required for a specific output path.
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
