"""Entry point for the dual AI assistant system.

Usage:
    python main.py --mode chat    # Launch Gradio UI
    python main.py --mode eval    # Run full evaluation suite
    python main.py --mode test    # Run pytest
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler
import logging

logging.basicConfig(
    level=logging.INFO,
    handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
    format="%(message)s",
    datefmt="[%X]",
)
logger = logging.getLogger(__name__)
console = Console()


def run_chat() -> None:
    """Initialise both assistants and launch the Gradio UI."""
    from config import config
    from src.assistants.oss_assistant import OSSAssistant
    from src.assistants.frontier_assistant import FrontierAssistant
    from src.tools.tool_registry import ToolRegistry
    from src.tools.web_search import WebSearchTool
    from src.ui.app import build_app

    console.print("[bold cyan]Starting Dual AI Assistant Chat UI...[/bold cyan]")
    registry = ToolRegistry()
    registry.register(WebSearchTool(max_results=3))

    oss = OSSAssistant(config=config, tool_registry=registry, user_id="oss")
    frontier = FrontierAssistant(config=config, tool_registry=registry, user_id="frontier")
    demo = build_app(oss, frontier)
    demo.launch(share=True)


def run_eval() -> None:
    """Run the full evaluation suite and generate an HTML report."""
    from config import config
    from src.assistants.oss_assistant import OSSAssistant
    from src.assistants.frontier_assistant import FrontierAssistant
    from src.evaluation.hallucination import HallucinationEvaluator
    from src.evaluation.bias_safety import SafetyEvaluator
    from src.evaluation.evaluator import EvalResult
    from src.evaluation.report_generator import ReportGenerator

    console.print("[bold cyan]Loading evaluation data...[/bold cyan]")

    eval_data_dir = Path(__file__).parent / "eval_data"
    factual_prompts: list[dict] = json.loads(
        (eval_data_dir / "factual_prompts.json").read_text(encoding="utf-8")
    )
    adversarial_prompts: list[dict] = json.loads(
        (eval_data_dir / "adversarial_prompts.json").read_text(encoding="utf-8")
    )
    bias_prompts: list[dict] = json.loads(
        (eval_data_dir / "bias_prompts.json").read_text(encoding="utf-8")
    )

    console.print("[bold cyan]Initialising assistants and evaluators...[/bold cyan]")
    oss = OSSAssistant(config=config)
    frontier = FrontierAssistant(config=config)
    hallucination_eval = HallucinationEvaluator(config=config)
    safety_eval = SafetyEvaluator(config=config)

    all_results: list[EvalResult] = []

    # -- Factual prompts -------------------------------------------------------
    console.print(f"\n[yellow]Running {len(factual_prompts)} factual prompts...[/yellow]")
    for prompt in factual_prompts:
        for assistant in (oss, frontier):
            assistant.reset()
            response = assistant.chat(prompt["prompt"])
            result = hallucination_eval.evaluate(prompt, response)
            all_results.append(result)
            console.print(
                f"  [{result.label.upper()}] {result.model_name} / {result.prompt_id} "
                f"score={result.score:.2f}"
            )

    # -- Adversarial prompts ---------------------------------------------------
    console.print(f"\n[yellow]Running {len(adversarial_prompts)} adversarial prompts...[/yellow]")
    for prompt in adversarial_prompts:
        for assistant in (oss, frontier):
            assistant.reset()
            response = assistant.chat(prompt["prompt"])
            result = safety_eval.evaluate(prompt, response)
            all_results.append(result)
            console.print(
                f"  [{result.label.upper()}] {result.model_name} / {result.prompt_id} "
                f"score={result.score:.2f}"
            )

    # -- Bias prompts ----------------------------------------------------------
    console.print(f"\n[yellow]Running {len(bias_prompts)} bias prompts...[/yellow]")
    for prompt in bias_prompts:
        for assistant in (oss, frontier):
            assistant.reset()
            response = assistant.chat(prompt["prompt"])
            result = safety_eval.evaluate(prompt, response)
            all_results.append(result)
            console.print(
                f"  [{result.label.upper()}] {result.model_name} / {result.prompt_id} "
                f"score={result.score:.2f}"
            )

    console.print(f"\n[bold green]Evaluation complete. {len(all_results)} results collected.[/bold green]")

    generator = ReportGenerator(output_dir="outputs")
    report_path = generator.generate(all_results)
    console.print(f"\n[bold green]Report saved to: {report_path}[/bold green]")


def run_tests() -> None:
    """Run the pytest test suite and print results."""
    import pytest

    console.print("[bold cyan]Running pytest...[/bold cyan]")
    exit_code = pytest.main(
        [
            "tests/",
            "-v",
            "--tb=short",
            "--no-header",
        ]
    )
    if exit_code == 0:
        console.print("[bold green]All tests passed.[/bold green]")
    else:
        console.print(f"[bold red]Tests failed (exit code {exit_code}).[/bold red]")
    sys.exit(int(exit_code))


def main() -> None:
    parser = argparse.ArgumentParser(description="Dual AI Assistant System")
    parser.add_argument(
        "--mode",
        choices=["chat", "eval", "test"],
        default="chat",
        help="Operating mode: chat (UI), eval (evaluation suite), test (pytest)",
    )
    args = parser.parse_args()

    if args.mode == "chat":
        run_chat()
    elif args.mode == "eval":
        run_eval()
    elif args.mode == "test":
        run_tests()


if __name__ == "__main__":
    main()
