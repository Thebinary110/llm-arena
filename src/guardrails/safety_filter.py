"""Real-time content moderation using LlamaGuard-style classification via Groq.

Uses llama-3.1-8b-instant with a LlamaGuard-format system prompt because the
original llama-guard-3-8b model was decommissioned by Groq. The response format
(safe / unsafe\nS1,S9) and all category codes are preserved identically.

Responsibility: classify text and return a SafetyResult.
This module never modifies or blocks text -- that decision belongs to the caller.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from groq import Groq, APIError, APITimeoutError, APIConnectionError
from rich.logging import RichHandler

from config import config

logging.basicConfig(handlers=[RichHandler(rich_tracebacks=True)], level=logging.INFO)
logger = logging.getLogger(__name__)

# LlamaGuard-3 category code -> human-readable name (all 14 categories).
_LLAMAGUARD_CATEGORIES: dict[str, str] = {
    "S1":  "Violent Crimes",
    "S2":  "Non-Violent Crimes",
    "S3":  "Sex-Related Crimes",
    "S4":  "Child Sexual Exploitation",
    "S5":  "Defamation",
    "S6":  "Specialized Advice",
    "S7":  "Privacy Violations",
    "S8":  "Intellectual Property",
    "S9":  "Indiscriminate Weapons",
    "S10": "Hate Speech",
    "S11": "Suicide and Self-Harm",
    "S12": "Sexual Content",
    "S13": "Elections",
    "S14": "Code Interpreter Abuse",
}

# System prompt that instructs the underlying LLM to behave exactly like LlamaGuard-3.
# Response must be "safe" or "unsafe\nS1" (or multiple codes like "unsafe\nS1,S9").
_SYSTEM_PROMPT = (
    "You are a content safety classifier. Analyze the user message and determine "
    "if it violates any of the following harm categories.\n\n"
    "Categories:\n"
    "S1: Violent Crimes - content enabling, encouraging, or excusing violence against people or animals\n"
    "S2: Non-Violent Crimes - content enabling illegal activities like theft, fraud, or trafficking\n"
    "S3: Sex-Related Crimes - sexual assault, harassment, or exploitation\n"
    "S4: Child Sexual Exploitation - any sexual content involving minors\n"
    "S5: Defamation - false statements materially damaging to reputations\n"
    "S6: Specialized Advice - dangerous medical, legal, or financial advice\n"
    "S7: Privacy Violations - unauthorized sharing of personal information\n"
    "S8: Intellectual Property - copyright or trademark violations\n"
    "S9: Indiscriminate Weapons - instructions for weapons capable of mass casualties (CBRN)\n"
    "S10: Hate Speech - content attacking groups based on protected characteristics\n"
    "S11: Suicide and Self-Harm - content encouraging or instructing self-harm\n"
    "S12: Sexual Content - explicit sexual content\n"
    "S13: Elections - false or misleading information about elections\n"
    "S14: Code Interpreter Abuse - malicious code, cyberattacks, or exploitation\n\n"
    "Respond with ONLY one of these two formats, nothing else:\n"
    '  "safe"  -- if the content is safe\n'
    '  "unsafe\\nS1"  -- if unsafe, where S1 is replaced with the applicable code(s) separated by commas\n\n'
    "Do not explain. Do not add any other text. Output only safe or unsafe followed by codes."
)

# ---------------------------------------------------------------------------
# Keyword filter patterns -- compiled once at module load.
# Categories catch context-dependent harmful content that LLM classifiers may
# miss when content is framed as educational or hypothetical.
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
    """Content moderation result from LlamaGuard-style classifier and/or keyword filter."""

    is_toxic: bool
    toxicity_score: float
    categories: dict[str, float] = field(default_factory=dict)
    flagged_categories: list[str] = field(default_factory=list)


class SafetyFilter:
    """Content moderation using LlamaGuard-style classification via Groq.

    The underlying model (config.LLAMAGUARD_MODEL) is prompted with a LlamaGuard-3
    system prompt so responses follow the "safe" / "unsafe\nS1,S9" format.
    The threshold parameter is kept for interface compatibility but is not used
    in is_toxic determination -- the classifier returns binary verdicts, not scores.
    hard_filter() runs as a second layer inside check() after the API call.
    """

    def __init__(self, threshold: float = 0.7) -> None:
        # threshold kept for interface compatibility only; not used in this path
        # because the classifier returns binary safe/unsafe, not a float score.
        self.threshold = threshold
        self._client: Groq | None = None
        if not config.GROQ_API_KEY:
            logger.warning("GROQ_API_KEY is not configured -- LlamaGuard moderation disabled.")
        else:
            self._client = Groq(api_key=config.GROQ_API_KEY)

    def hard_filter(self, text: str) -> str | None:
        """Keyword-based filter for content the LLM classifier may miss.

        Returns the matched category name if a keyword is found, else None.
        Also called inside check() as a second layer after the API result.
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
        """Classify text via LlamaGuard-style API call and return a SafetyResult.

        Also applies keyword hard filter as a second layer.
        Returns a safe (non-toxic) SafetyResult on any failure so the system
        degrades gracefully when the API is unavailable.
        """
        # Step 1: empty input
        if not text or not text.strip():
            return SafetyResult(is_toxic=False, toxicity_score=0.0)

        # Step 2: no API key configured
        if self._client is None:
            logger.warning("GROQ_API_KEY not configured -- returning safe default.")
            return SafetyResult(is_toxic=False, toxicity_score=0.0)

        # Step 3: call the LlamaGuard-style classifier via Groq
        try:
            response = self._client.chat.completions.create(
                model=config.LLAMAGUARD_MODEL,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": text},
                ],
                max_tokens=20,
                temperature=0,
            )
            response_text = response.choices[0].message.content.strip()
        except (APIError, APITimeoutError, APIConnectionError) as exc:
            logger.error("Groq LlamaGuard API error: %s", exc)
            return SafetyResult(is_toxic=False, toxicity_score=0.0)
        except Exception as exc:
            logger.error("Unexpected LlamaGuard error: %s", exc)
            return SafetyResult(is_toxic=False, toxicity_score=0.0)

        # Step 4: parse response -- "safe" or "unsafe\nS1,S9" etc.
        # Build base categories dict with all S-codes defaulting to 0.0
        categories: dict[str, float] = {
            name: 0.0 for name in _LLAMAGUARD_CATEGORIES.values()
        }
        flagged_cats: list[str] = []
        is_toxic = False
        toxicity_score = 0.0

        logger.info("LlamaGuard raw response: %r", response_text)

        if response_text.lower().startswith("unsafe"):
            is_toxic = True
            toxicity_score = 1.0
            # Extract the portion after "unsafe" -- models may use \n, \, space,
            # or no separator before the S-codes.
            after_unsafe = response_text[6:]  # slice past "unsafe"
            # Split on any non-alphanumeric characters to extract S-codes
            raw_codes = re.split(r"[^A-Za-z0-9]+", after_unsafe)
            for code in raw_codes:
                code = code.strip()
                if code in _LLAMAGUARD_CATEGORIES:
                    cat_name = _LLAMAGUARD_CATEGORIES[code]
                    categories[cat_name] = 1.0
                    if cat_name not in flagged_cats:
                        flagged_cats.append(cat_name)

        # Step 5: keyword hard filter as second layer -- overrides a safe verdict
        kw_hit = self.hard_filter(text)
        if kw_hit is not None:
            is_toxic = True
            if toxicity_score < 1.0:
                toxicity_score = 1.0
            kw_category = f"keyword:{kw_hit}"
            if kw_category not in flagged_cats:
                flagged_cats.append(kw_category)

        # Step 6: return combined result
        return SafetyResult(
            is_toxic=is_toxic,
            toxicity_score=toxicity_score,
            categories=categories,
            flagged_categories=flagged_cats,
        )
