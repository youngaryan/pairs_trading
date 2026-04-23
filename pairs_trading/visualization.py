from __future__ import annotations

from html import escape
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .backtesting import ExperimentResult, json_ready


class ExperimentVisualizer:
    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _save_current_figure(self, filename: str) -> Path:
        path = self.output_dir / filename
        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        return path

    def plot_equity_curve(self, result: ExperimentResult) -> Path:
        frame = result.equity_curve.sort_index()
        plt.figure(figsize=(12, 5))
        plt.plot(frame.index, frame["equity_curve"], color="#155eef", linewidth=2, label="Net Equity")
        plt.axhline(1.0, color="#111827", linestyle="--", linewidth=1, alpha=0.5)
        plt.title("Walk-Forward Equity Curve")
        plt.xlabel("Date")
        plt.ylabel("Equity")
        plt.legend()
        plt.grid(alpha=0.25)
        return self._save_current_figure("equity_curve.png")

    def plot_drawdown(self, result: ExperimentResult) -> Path:
        frame = result.equity_curve.sort_index()
        equity = frame["equity_curve"].fillna(1.0)
        drawdown = equity / equity.cummax() - 1.0

        plt.figure(figsize=(12, 4))
        plt.fill_between(drawdown.index, drawdown.values, 0, color="#dc2626", alpha=0.35)
        plt.plot(drawdown.index, drawdown.values, color="#991b1b", linewidth=1.5)
        plt.title("Drawdown")
        plt.xlabel("Date")
        plt.ylabel("Drawdown")
        plt.grid(alpha=0.25)
        return self._save_current_figure("drawdown.png")

    def plot_exposure_and_costs(self, result: ExperimentResult) -> Path:
        frame = result.equity_curve.sort_index()
        plt.figure(figsize=(12, 5))
        if "gross_exposure" in frame.columns:
            plt.plot(frame.index, frame["gross_exposure"], label="Gross Exposure", color="#0f766e")
        if "short_exposure" in frame.columns:
            plt.plot(frame.index, frame["short_exposure"], label="Short Exposure", color="#7c3aed")
        if "total_cost" in frame.columns:
            plt.plot(frame.index, frame["total_cost"], label="Total Cost", color="#ea580c", alpha=0.8)
        if "turnover" in frame.columns:
            plt.plot(frame.index, frame["turnover"], label="Turnover", color="#0369a1", alpha=0.7)
        plt.title("Exposure, Costs, and Turnover")
        plt.xlabel("Date")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(alpha=0.25)
        return self._save_current_figure("exposure_costs.png")

    def plot_strategy_weights(self, result: ExperimentResult) -> Path | None:
        frame = result.equity_curve.sort_index()
        weight_columns = [column for column in frame.columns if column.startswith("weight_")]
        if not weight_columns:
            return None

        weight_matrix = frame[weight_columns].T
        if weight_matrix.empty:
            return None

        plt.figure(figsize=(12, max(3, len(weight_columns) * 0.5)))
        vmax = max(0.1, float(np.nanmax(np.abs(weight_matrix.values))))
        plt.imshow(weight_matrix.values, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
        tick_positions = np.linspace(0, len(frame.index) - 1, num=min(8, len(frame.index)), dtype=int)
        plt.xticks(tick_positions, [frame.index[i].strftime("%Y-%m-%d") for i in tick_positions], rotation=45, ha="right")
        plt.yticks(range(len(weight_columns)), [column.replace("weight_", "") for column in weight_columns])
        plt.title("Strategy Weight Heatmap")
        plt.colorbar(label="Weight")
        return self._save_current_figure("strategy_weights.png")

    def plot_sentiment_overlay(self, result: ExperimentResult) -> Path | None:
        frame = result.equity_curve.sort_index()
        if "sentiment_strength" not in frame.columns and "sentiment_confidence" not in frame.columns:
            return None

        plt.figure(figsize=(12, 4))
        if "sentiment_strength" in frame.columns:
            plt.plot(frame.index, frame["sentiment_strength"], label="Sentiment Strength", color="#16a34a")
        if "sentiment_confidence" in frame.columns:
            plt.plot(frame.index, frame["sentiment_confidence"], label="Sentiment Confidence", color="#ca8a04")
        plt.axhline(0.0, color="#111827", linestyle="--", linewidth=1, alpha=0.4)
        plt.title("Sentiment Overlay")
        plt.xlabel("Date")
        plt.ylabel("Score")
        plt.legend()
        plt.grid(alpha=0.25)
        return self._save_current_figure("sentiment_overlay.png")

    def plot_fold_sharpe_and_return(self, result: ExperimentResult) -> Path:
        metrics = result.fold_metrics.copy()
        plt.figure(figsize=(10, 4))
        x = np.arange(len(metrics))
        plt.bar(x - 0.2, metrics["total_return"], width=0.4, label="Total Return", color="#155eef")
        plt.bar(x + 0.2, metrics["sharpe"], width=0.4, label="Sharpe", color="#16a34a")
        plt.xticks(x, [f"Fold {int(value)}" for value in metrics["fold"]], rotation=0)
        plt.title("Per-Fold Return and Sharpe")
        plt.legend()
        plt.grid(axis="y", alpha=0.25)
        return self._save_current_figure("fold_metrics.png")

    def plot_latest_pair_selection(self, result: ExperimentResult) -> Path | None:
        if not result.diagnostics:
            return None

        latest = result.diagnostics[-1].get("diagnostics", {})
        selected_pairs = latest.get("selected_pairs", [])
        if not selected_pairs:
            return None

        frame = pd.DataFrame(selected_pairs)
        score_column = "Adjusted_Rank_Score" if "Adjusted_Rank_Score" in frame.columns else "Rank_Score"
        if score_column not in frame.columns:
            return None

        labels = [f"{row['Ticker_1']}/{row['Ticker_2']}" for _, row in frame.iterrows()]
        colors = "#16a34a"

        plt.figure(figsize=(10, max(3, len(labels) * 0.6)))
        plt.barh(labels, frame[score_column], color=colors)
        plt.title(f"Latest Fold Selected Pairs ({score_column})")
        plt.xlabel(score_column)
        plt.grid(axis="x", alpha=0.25)
        return self._save_current_figure("latest_pair_selection.png")

    def build_html_report(self, result: ExperimentResult, chart_paths: list[Path]) -> Path:
        summary_items = "".join(
            f"<tr><th>{escape(str(key))}</th><td>{escape(str(value))}</td></tr>"
            for key, value in json_ready(result.summary).items()
        )
        charts = "".join(
            f"<section><h2>{escape(path.stem.replace('_', ' ').title())}</h2>"
            f"<img src='{escape(path.name)}' style='max-width: 100%; border: 1px solid #d1d5db; border-radius: 8px;' /></section>"
            for path in chart_paths
        )
        diagnostics_preview = json.dumps(json_ready(result.diagnostics[-1] if result.diagnostics else {}), indent=2)

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>{escape(result.experiment_id)} Report</title>
  <style>
    body {{ font-family: Georgia, serif; margin: 2rem; color: #111827; background: #f8fafc; }}
    h1, h2 {{ color: #0f172a; }}
    table {{ border-collapse: collapse; width: 100%; max-width: 900px; background: white; }}
    th, td {{ border: 1px solid #e5e7eb; padding: 0.6rem; text-align: left; }}
    th {{ width: 260px; background: #f1f5f9; }}
    section {{ margin: 1.5rem 0; }}
    pre {{ background: #0f172a; color: #e2e8f0; padding: 1rem; overflow: auto; border-radius: 8px; }}
  </style>
</head>
<body>
  <h1>{escape(result.experiment_id)}</h1>
  <section>
    <h2>Summary</h2>
    <table>{summary_items}</table>
  </section>
  {charts}
  <section>
    <h2>Latest Diagnostics</h2>
    <pre>{escape(diagnostics_preview)}</pre>
  </section>
</body>
</html>
"""

        report_path = self.output_dir / "report.html"
        report_path.write_text(html, encoding="utf-8")
        return report_path

    def create_dashboard(self, result: ExperimentResult) -> dict[str, Path]:
        chart_paths: list[Path] = []
        generated: dict[str, Path] = {}

        for name, method in (
            ("equity_curve", self.plot_equity_curve),
            ("drawdown", self.plot_drawdown),
            ("exposure_costs", self.plot_exposure_and_costs),
            ("fold_metrics", self.plot_fold_sharpe_and_return),
        ):
            path = method(result)
            chart_paths.append(path)
            generated[name] = path

        for name, method in (
            ("strategy_weights", self.plot_strategy_weights),
            ("sentiment_overlay", self.plot_sentiment_overlay),
            ("latest_pair_selection", self.plot_latest_pair_selection),
        ):
            path = method(result)
            if path is not None:
                chart_paths.append(path)
                generated[name] = path

        generated["report"] = self.build_html_report(result, chart_paths)
        return generated
