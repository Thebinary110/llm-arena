"""Converts EvalResult objects into Evidently HTML reports and Rich console tables."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import Any

import pandas as pd
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from src.evaluation.evaluator import EvalResult

logging.basicConfig(handlers=[RichHandler(rich_tracebacks=True)], level=logging.INFO)
logger = logging.getLogger(__name__)
console = Console()


class ReportGenerator:
    """Consumes a list of EvalResult objects and produces reports.

    Responsibility: render data only.  It does not run evaluations,
    call APIs, or manage assistants.
    """

    def __init__(self, output_dir: str = "outputs") -> None:
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def generate(self, results: list[EvalResult]) -> str:
        """Build HTML report + JSON dump from results; return path to HTML file.

        Steps:
          1. Convert results to DataFrame
          2. Compute summary statistics per model per category
          3. Generate Evidently HTML report
          4. Save raw results as JSON
          5. Print Rich summary table to console
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        html_path = os.path.join(self.output_dir, f"eval_report_{timestamp}.html")
        json_path = os.path.join(self.output_dir, f"eval_results_{timestamp}.json")

        df = self._to_dataframe(results)

        self._save_json(results, json_path)
        self._print_rich_summary(df)
        self._generate_html_report(df, html_path)

        logger.info("HTML report saved to: %s", html_path)
        logger.info("JSON results saved to: %s", json_path)
        return html_path

    # ── private helpers ────────────────────────────────────────────────────────

    def _to_dataframe(self, results: list[EvalResult]) -> pd.DataFrame:
        """Convert list of EvalResult to a flat pandas DataFrame."""
        rows = [
            {
                "prompt_id": r.prompt_id,
                "category": r.category,
                "model_name": r.model_name,
                "prompt": r.prompt,
                "response": r.response,
                "score": r.score,
                "label": r.label,
                "reasoning": r.reasoning,
                "latency_ms": r.latency_ms,
                "is_toxic": r.is_toxic,
                "toxicity_score": r.toxicity_score,
            }
            for r in results
        ]
        return pd.DataFrame(rows)

    def _save_json(self, results: list[EvalResult], path: str) -> None:
        """Persist raw EvalResult list as JSON."""
        data = [
            {
                "prompt_id": r.prompt_id,
                "category": r.category,
                "model_name": r.model_name,
                "prompt": r.prompt,
                "response": r.response,
                "score": r.score,
                "label": r.label,
                "reasoning": r.reasoning,
                "latency_ms": r.latency_ms,
                "is_toxic": r.is_toxic,
                "toxicity_score": r.toxicity_score,
            }
            for r in results
        ]
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)

    def _print_rich_summary(self, df: pd.DataFrame) -> None:
        """Print a summary table of scores per model per category to the console."""
        if df.empty:
            console.print("[yellow]No results to summarise.[/yellow]")
            return

        summary = (
            df.groupby(["model_name", "category"])
            .agg(
                mean_score=("score", "mean"),
                pass_rate=("label", lambda s: (s == "pass").mean()),
                avg_latency_ms=("latency_ms", "mean"),
                n=("score", "count"),
            )
            .reset_index()
        )

        table = Table(title="Evaluation Summary", show_lines=True)
        table.add_column("Model", style="cyan", no_wrap=True)
        table.add_column("Category", style="magenta")
        table.add_column("Mean Score", justify="right")
        table.add_column("Pass Rate", justify="right")
        table.add_column("Avg Latency (ms)", justify="right")
        table.add_column("N", justify="right")

        for _, row in summary.iterrows():
            score_color = (
                "green" if row["mean_score"] >= 0.7 else
                "yellow" if row["mean_score"] >= 0.4 else "red"
            )
            table.add_row(
                str(row["model_name"]),
                str(row["category"]),
                f"[{score_color}]{row['mean_score']:.2f}[/{score_color}]",
                f"{row['pass_rate']:.0%}",
                f"{row['avg_latency_ms']:.0f}",
                str(int(row["n"])),
            )

        console.print(table)

    def _generate_html_report(self, df: pd.DataFrame, path: str) -> None:
        """Generate an HTML report using Evidently when available; fall back to pandas."""
        try:
            self._generate_evidently_report(df, path)
        except Exception as exc:
            logger.warning("Evidently report failed (%s); falling back to pandas HTML.", exc)
            self._generate_pandas_html_report(df, path)

    def _generate_evidently_report(self, df: pd.DataFrame, path: str) -> None:
        """Use Evidently to produce a structured HTML report."""
        from evidently.report import Report
        from evidently.metric_preset import DataDriftPreset
        from evidently import ColumnMapping

        if df.empty or "model_name" not in df.columns:
            raise ValueError("DataFrame is empty or missing model_name column.")

        models = df["model_name"].unique()
        if len(models) < 2:
            raise ValueError("Need at least two models for Evidently drift comparison.")

        model_a, model_b = models[0], models[1]
        df_a = df[df["model_name"] == model_a][["score", "latency_ms", "toxicity_score"]].reset_index(drop=True)
        df_b = df[df["model_name"] == model_b][["score", "latency_ms", "toxicity_score"]].reset_index(drop=True)

        # Pad to equal length so Evidently can compare
        min_len = min(len(df_a), len(df_b))
        df_a = df_a.iloc[:min_len]
        df_b = df_b.iloc[:min_len]

        column_mapping = ColumnMapping(numerical_features=["score", "latency_ms", "toxicity_score"])
        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=df_a, current_data=df_b, column_mapping=column_mapping)
        report.save_html(path)

    def _generate_pandas_html_report(self, df: pd.DataFrame, path: str) -> None:
        """Fallback: write a styled pandas HTML table as the report."""
        summary = (
            df.groupby(["model_name", "category"])
            .agg(
                mean_score=("score", "mean"),
                pass_rate=("label", lambda s: (s == "pass").mean()),
                avg_latency_ms=("latency_ms", "mean"),
                toxic_responses=("is_toxic", "sum"),
                n=("score", "count"),
            )
            .reset_index()
        )

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Dual AI Evaluation Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 40px; }}
    h1 {{ color: #2c3e50; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
    th {{ background: #2c3e50; color: white; padding: 10px; text-align: left; }}
    td {{ padding: 8px; border-bottom: 1px solid #ddd; }}
    tr:hover {{ background-color: #f5f5f5; }}
  </style>
</head>
<body>
  <h1>Dual AI Evaluation Report</h1>
  <h2>Summary Statistics</h2>
  {summary.to_html(index=False, float_format="{:.3f}".format)}
  <h2>Full Results</h2>
  {df[["prompt_id","category","model_name","score","label","latency_ms","toxicity_score"]].to_html(index=False, float_format="{:.3f}".format)}
</body>
</html>"""

        with open(path, "w", encoding="utf-8") as fh:
            fh.write(html)
