from __future__ import annotations

from pathlib import Path
import unittest

import pandas as pd

from pairs_trading.backtesting import ExperimentResult
from tests.common import fresh_test_dir
from pairs_trading.visualization import ExperimentVisualizer


class VisualizationTests(unittest.TestCase):
    def test_visualizer_creates_dashboard_files(self) -> None:
        output_dir = fresh_test_dir("artifacts/test_visuals")
        index = pd.date_range("2024-01-01", periods=5, freq="D")

        equity_curve = pd.DataFrame(
            {
                "equity_curve": [1.0, 1.01, 1.02, 1.015, 1.03],
                "gross_equity_curve": [1.0, 1.012, 1.024, 1.020, 1.037],
                "net_return": [0.0, 0.01, 0.0099, -0.0049, 0.0148],
                "gross_exposure": [0.8, 0.9, 1.0, 0.95, 0.85],
                "short_exposure": [0.3, 0.35, 0.4, 0.38, 0.32],
                "risk_gross_exposure": [0.8, 0.85, 0.9, 0.88, 0.82],
                "risk_net_exposure": [0.2, 0.25, 0.3, 0.28, 0.22],
                "risk_scale": [1.0, 0.95, 0.92, 0.97, 1.0],
                "risk_flag": [0, 1, 1, 0, 0],
                "total_cost": [0.001, 0.0012, 0.0015, 0.0011, 0.0010],
                "execution_cost": [0.0004, 0.0005, 0.0007, 0.0005, 0.0004],
                "impact_cost": [0.0002, 0.0002, 0.0003, 0.0002, 0.0002],
                "latency_cost": [0.0001, 0.0001, 0.00015, 0.0001, 0.0001],
                "borrow_cost": [0.0001, 0.0001, 0.00015, 0.0001, 0.0001],
                "strategy_cost": [0.0002, 0.0003, 0.0002, 0.0002, 0.0002],
                "participation_rate": [0.10, 0.15, 0.18, 0.12, 0.10],
                "turnover": [0.2, 0.3, 0.25, 0.2, 0.15],
                "sentiment_strength": [0.0, 0.1, 0.2, 0.15, 0.05],
                "sentiment_confidence": [0.0, 0.7, 0.8, 0.75, 0.7],
                "weight_AAA_BBB": [0.2, 0.25, 0.3, 0.25, 0.2],
                "weight_CCC_DDD": [-0.2, -0.25, -0.3, -0.2, -0.15],
            },
            index=index,
        )
        fold_metrics = pd.DataFrame(
            {
                "fold": [1, 2],
                "total_return": [0.05, 0.03],
                "sharpe": [1.2, 0.9],
            }
        )
        diagnostics = [
            {
                "fold": 2,
                "diagnostics": {
                    "selected_pairs": [
                        {"Ticker_1": "AAA", "Ticker_2": "BBB", "Adjusted_Rank_Score": 0.8},
                        {"Ticker_1": "CCC", "Ticker_2": "DDD", "Adjusted_Rank_Score": 0.7},
                    ]
                },
            }
        ]
        result = ExperimentResult(
            experiment_id="visual_test",
            summary={"total_return": 0.08, "sharpe": 1.05, "dsr": 0.74, "pbo": 0.18, "strategy": "stat_arb", "folds": 2},
            fold_metrics=fold_metrics,
            equity_curve=equity_curve,
            diagnostics=diagnostics,
            artifact_dir=Path(output_dir),
            validation={"psr": 0.81, "dsr": 0.74, "pbo": 0.18, "trial_count": 3},
        )

        visualizer = ExperimentVisualizer(output_dir)
        artifacts = visualizer.create_dashboard(result)

        expected = {
            "equity_curve",
            "drawdown",
            "exposure_costs",
            "execution_diagnostics",
            "fold_metrics",
            "strategy_weights",
            "component_summary",
            "sentiment_overlay",
            "validation_snapshot",
            "latest_pair_selection",
            "report",
        }
        self.assertTrue(expected.issubset(set(artifacts)))
        for path in artifacts.values():
            self.assertTrue(Path(path).exists())

        report_text = Path(artifacts["report"]).read_text(encoding="utf-8")
        self.assertIn("Quant Research Dashboard", report_text)
        self.assertIn("Validation", report_text)
        self.assertIn("Execution & Risk", report_text)
        self.assertIn("Selected Pairs", report_text)


if __name__ == "__main__":
    unittest.main()
