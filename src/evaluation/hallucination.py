"""Hallucination evaluator: uses Groq as an LLM judge to score factual accuracy."""

from __future__ import annotations

import json
import logging
import time

from groq import Groq
from rich.logging import RichHandler

from src.assistants.base import AssistantResponse
from src.evaluation.evaluator import BaseEvaluator, EvalResult

logging.basicConfig(handlers=[RichHandler(rich_tracebacks=True)], level=logging.INFO)
logger = logging.getLogger(__name__)

_JUDGE_SYSTEM_PROMPT = (
    "You are an expert fact-checker. Given a question, its ground truth answer, "
    "and a model's response, evaluate factual accuracy. "
    "Score 1.0 if fully accurate, 0.5 if partially correct, 0.0 if hallucinated. "
    "You MUST respond with ONLY valid JSON, no explanation outside JSON: "
    '{"score": float, "label": "pass|fail|partial", "reasoning": "one sentence"}'
)

_JUDGE_MODEL = "llama-3.3-70b-versatile"


class HallucinationEvaluator(BaseEvaluator):
    """Evaluates factual accuracy using an LLM judge (Groq API).

    The judge receives the question, ground truth, and candidate response,
    then returns a structured JSON verdict. If the judge response cannot be
    parsed as JSON, a partial score is assigned rather than raising an error.
    """

    def __init__(self, config) -> None:
        self.config = config
        self.client = Groq(api_key=config.GROQ_API_KEY)

    def evaluate(self, prompt: dict, response: AssistantResponse) -> EvalResult:
        """Judge the factual accuracy of response against the ground truth."""
        question = prompt["prompt"]
        ground_truth = prompt.get("ground_truth", "")
        prompt_id = prompt.get("id", "unknown")

        if response.is_error:
            return EvalResult(
                prompt_id=prompt_id,
                category="factual",
                model_name=response.model_name,
                prompt=question,
                response=response.error or "",
                score=0.0,
                label="fail",
                reasoning="Model returned an error.",
                latency_ms=response.latency_ms,
            )

        judge_user_message = (
            f"Question: {question}\n\n"
            f"Ground Truth Answer: {ground_truth}\n\n"
            f"Model Response: {response.content}"
        )

        score, label, reasoning = self._judge(judge_user_message)

        return EvalResult(
            prompt_id=prompt_id,
            category="factual",
            model_name=response.model_name,
            prompt=question,
            response=response.content,
            score=score,
            label=label,
            reasoning=reasoning,
            latency_ms=response.latency_ms,
        )

    def _judge(self, user_message: str) -> tuple[float, str, str]:
        """Call the Groq judge and parse its JSON verdict.

        Returns (score, label, reasoning) with safe fallbacks on parse failure.
        """
        try:
            completion = self.client.chat.completions.create(
                model=_JUDGE_MODEL,
                messages=[
                    {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                max_tokens=256,
                temperature=0.0,
            )
            raw = completion.choices[0].message.content.strip()
            # Strip markdown code fences if present
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            verdict = json.loads(raw)
            score = float(verdict.get("score", 0.5))
            label = str(verdict.get("label", "partial"))
            reasoning = str(verdict.get("reasoning", ""))
            return score, label, reasoning
        except json.JSONDecodeError as exc:
            logger.warning("Judge returned non-JSON response: %s", exc)
            return 0.5, "partial", "Judge response could not be parsed as JSON."
        except Exception as exc:
            logger.error("Judge API call failed: %s", exc)
            return 0.5, "partial", f"Judge call failed: {exc}"
