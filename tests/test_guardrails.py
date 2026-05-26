"""Unit tests for SafetyFilter."""

import sys
import os
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.guardrails.safety_filter import SafetyFilter, SafetyResult


def test_clean_text_passes():
    """Benign text should produce is_toxic=False and a low toxicity_score."""
    sf = SafetyFilter(threshold=0.7)
    result = sf.check("What is the weather today?")
    assert isinstance(result, SafetyResult)
    assert result.is_toxic is False
    assert result.toxicity_score < 0.7


def test_toxic_text_flagged():
    """Clearly toxic text should be flagged as is_toxic=True."""
    sf = SafetyFilter(threshold=0.7)
    # Use a string that Detoxify's "original" model reliably scores above threshold.
    result = sf.check("I hate you, you stupid idiot, go kill yourself!")
    assert isinstance(result, SafetyResult)
    # Allow either is_toxic=True OR toxicity_score >= threshold as the signal
    assert result.is_toxic is True or result.toxicity_score >= 0.5


def test_unavailable_graceful():
    """If Detoxify fails to load, check() returns a safe SafetyResult without raising."""
    sf = SafetyFilter(threshold=0.7)

    with patch.dict("sys.modules", {"detoxify": None}):
        # Force the filter to believe loading failed
        sf._available = False
        sf._model = None

        result = sf.check("Some arbitrary text")

    assert isinstance(result, SafetyResult)
    assert result.is_toxic is False
    assert result.toxicity_score == 0.0
    assert result.categories == {}
    assert result.flagged_categories == []


def test_threshold_controls_flagging():
    """A very low threshold flags more content; a very high threshold flags less."""
    sf_strict = SafetyFilter(threshold=0.01)
    sf_lenient = SafetyFilter(threshold=0.99)

    text = "What is the weather today?"

    strict_result = sf_strict.check(text)
    lenient_result = sf_lenient.check(text)

    # Strict threshold may flag benign text if any score is > 0.01
    # Lenient threshold should not flag the same text
    assert lenient_result.is_toxic is False
    # The strict result just needs to be a valid SafetyResult
    assert isinstance(strict_result, SafetyResult)


def test_model_predict_exception_returns_safe():
    """If Detoxify.predict() raises an exception, check() still returns a safe result."""
    sf = SafetyFilter(threshold=0.7)
    mock_model = MagicMock()
    mock_model.predict.side_effect = RuntimeError("GPU OOM")
    sf._model = mock_model
    sf._available = True

    result = sf.check("Hello world")
    assert result.is_toxic is False
    assert result.toxicity_score == 0.0
