from __future__ import annotations

from types import SimpleNamespace
import time
import unittest
from unittest.mock import patch

import pandas as pd

from tests.common import fresh_test_dir

try:
    from fastapi.testclient import TestClient
except ImportError:  # pragma: no cover - optional backend dependency
    TestClient = None


@unittest.skipIf(TestClient is None, "FastAPI backend dependencies are not installed.")
class BackendBacktestTests(unittest.TestCase):
    def test_backtest_job_routes_submit_and_complete(self) -> None:
        from pairs_trading.backend.app import create_app
        from pairs_trading.backend.config import BackendSettings

        workspace = fresh_test_dir("artifacts/test_backend_backtests")
        artifact_dir = workspace / "experiments" / "fake"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        fake_result = SimpleNamespace(
            artifact_dir=artifact_dir,
            fold_metrics=pd.DataFrame({"fold": [1], "sharpe": [1.2]}),
            equity_curve=pd.DataFrame({"net_return": [0.01, 0.02]}, index=pd.date_range("2024-01-01", periods=2)),
        )

        app = create_app(
            BackendSettings(
                paper_state_dir=workspace / "state",
                paper_artifact_root=workspace / "paper_runs",
                backtest_artifact_root=workspace / "backtest_experiments",
                backtest_job_state_dir=workspace / "backtest_jobs",
                price_cache_dir=workspace / "cache",
                default_paper_config=workspace / "missing.json",
            )
        )
        client = TestClient(app)

        templates = client.get("/api/backtests/templates")
        self.assertEqual(templates.status_code, 200)
        self.assertGreaterEqual(len(templates.json()), 3)

        with patch(
            "pairs_trading.backend.services.run_directional_pipeline",
            return_value={
                "summary": {"annualized_return": 0.12, "sharpe": 1.2, "max_drawdown": -0.05, "avg_turnover": 0.2, "folds": 4},
                "validation": {"dsr": 0.74, "pbo": 0.2},
                "visuals": {"report": str(artifact_dir / "report.html")},
                "result": fake_result,
            },
        ) as run_directional:
            submitted = client.post(
                "/api/backtests/run",
                json={
                    "pipeline": "ema_cross",
                    "symbols": ["SPY", "QQQ"],
                    "start": "2021-01-01",
                    "end": "2022-01-01",
                    "experiment_name": "unit_ui_backtest",
                    "parameters": {"ema_fast_window": 10, "ema_slow_window": 35},
                },
            )
            self.assertEqual(submitted.status_code, 202)
            job_id = submitted.json()["id"]

            job_payload = submitted.json()
            for _ in range(40):
                job = client.get(f"/api/backtests/jobs/{job_id}")
                self.assertEqual(job.status_code, 200)
                job_payload = job.json()
                if job_payload["status"] in {"completed", "failed"}:
                    break
                time.sleep(0.05)

            self.assertEqual(job_payload["status"], "completed")
            self.assertEqual(job_payload["progress"], 1.0)
            self.assertEqual(job_payload["stage"], "completed")
            self.assertEqual(job_payload["result"]["summary"]["sharpe"], 1.2)
            self.assertEqual(job_payload["result"]["validation"]["pbo"], 0.2)
            self.assertEqual(job_payload["result"]["artifact_dir"], str(artifact_dir))
            self.assertEqual(job_payload["result"]["decision"]["verdict"], "paper_candidate")
            self.assertGreater(len(job_payload["result"]["equity_curve_points"]), 0)
            self.assertTrue((workspace / "backtest_jobs" / f"{job_id}.json").exists())
            run_directional.assert_called_once()

        jobs = client.get("/api/backtests/jobs")
        self.assertEqual(jobs.status_code, 200)
        self.assertEqual(jobs.json()[0]["id"], job_id)


if __name__ == "__main__":
    unittest.main()
