from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .market_data import CachedParquetProvider, YahooFinanceProvider
from .portfolio import PortfolioManager
from .research import PairScreenConfig
from .stat_arb import SectorStatArbPipeline, StatArbConfig
from .strategies import StrategyOutput, WalkForwardStrategy


@dataclass(frozen=True)
class CostModel:
    commission_bps: float = 1.0
    spread_bps: float = 1.0
    slippage_bps: float = 1.0
    borrow_bps_annual: float = 50.0

    @property
    def transaction_cost_rate(self) -> float:
        return (self.commission_bps + self.spread_bps + self.slippage_bps) / 10_000.0


@dataclass(frozen=True)
class WalkForwardConfig:
    train_bars: int
    test_bars: int
    step_bars: int | None = None
    bars_per_year: int = 252


@dataclass
class ExperimentResult:
    experiment_id: str
    summary: dict[str, Any]
    fold_metrics: pd.DataFrame
    equity_curve: pd.DataFrame
    diagnostics: list[dict[str, Any]]
    artifact_dir: Path


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [json_ready(item) for item in value]
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")
    if isinstance(value, pd.Series):
        return value.to_dict()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


class WalkForwardBacktester:
    def __init__(
        self,
        strategy: WalkForwardStrategy,
        prices: pd.DataFrame,
        config: WalkForwardConfig,
        cost_model: CostModel,
        experiment_root: str | Path = "artifacts/experiments",
    ) -> None:
        self.strategy = strategy
        self.prices = prices.sort_index()
        self.config = config
        self.cost_model = cost_model
        self.experiment_root = Path(experiment_root)
        self.experiment_root.mkdir(parents=True, exist_ok=True)

    def _fold_boundaries(self) -> list[tuple[int, int, int]]:
        boundaries: list[tuple[int, int, int]] = []
        step = self.config.step_bars or self.config.test_bars
        test_start = self.config.train_bars

        while test_start + self.config.test_bars <= len(self.prices):
            train_start = test_start - self.config.train_bars
            test_end = test_start + self.config.test_bars
            boundaries.append((train_start, test_start, test_end))
            test_start += step

        return boundaries

    def _apply_cost_model(self, output: StrategyOutput) -> pd.DataFrame:
        frame = output.frame.copy()
        output.validate(extra_columns=("gross_return",))

        if "turnover" not in frame.columns:
            frame["turnover"] = frame["position"].diff().abs().fillna(frame["position"].abs())
        frame["turnover"] = frame["turnover"].fillna(0.0)

        frame["execution_cost"] = frame["turnover"] * self.cost_model.transaction_cost_rate

        short_exposure = frame.get("short_exposure", frame["position"].shift(1).abs().fillna(0.0) * 0.5)
        frame["borrow_cost"] = short_exposure.fillna(0.0) * (
            self.cost_model.borrow_bps_annual / 10_000.0 / self.config.bars_per_year
        )
        frame["strategy_cost"] = frame["cost_estimate"].fillna(0.0)
        frame["total_cost"] = frame["execution_cost"] + frame["borrow_cost"] + frame["strategy_cost"]
        frame["net_return"] = frame["gross_return"].fillna(0.0) - frame["total_cost"]
        frame["equity_curve"] = (1.0 + frame["net_return"]).cumprod()
        return frame

    def _compute_metrics(self, frame: pd.DataFrame) -> dict[str, float]:
        if frame.empty:
            return {
                "bars": 0,
                "total_return": 0.0,
                "annualized_return": 0.0,
                "annualized_vol": 0.0,
                "sharpe": 0.0,
                "max_drawdown": 0.0,
                "avg_turnover": 0.0,
                "hit_rate": 0.0,
            }

        net_returns = frame["net_return"].fillna(0.0)
        equity_curve = (1.0 + net_returns).cumprod()
        total_return = float(equity_curve.iloc[-1] - 1.0)
        bars = len(frame)

        mean_return = float(net_returns.mean())
        vol = float(net_returns.std(ddof=0))
        annualized_return = float((1.0 + total_return) ** (self.config.bars_per_year / max(bars, 1)) - 1.0)
        annualized_vol = float(vol * np.sqrt(self.config.bars_per_year))
        sharpe = float(mean_return / vol * np.sqrt(self.config.bars_per_year)) if vol > 0 else 0.0

        rolling_max = equity_curve.cummax()
        drawdown = equity_curve / rolling_max - 1.0
        max_drawdown = float(drawdown.min())

        return {
            "bars": float(bars),
            "total_return": total_return,
            "annualized_return": annualized_return,
            "annualized_vol": annualized_vol,
            "sharpe": sharpe,
            "max_drawdown": max_drawdown,
            "avg_turnover": float(frame["turnover"].mean()),
            "hit_rate": float((net_returns > 0).mean()),
        }

    def _save_artifacts(
        self,
        experiment_id: str,
        summary: dict[str, Any],
        fold_metrics: pd.DataFrame,
        equity_curve: pd.DataFrame,
        diagnostics: list[dict[str, Any]],
    ) -> Path:
        artifact_dir = self.experiment_root / experiment_id
        artifact_dir.mkdir(parents=True, exist_ok=True)

        with (artifact_dir / "summary.json").open("w", encoding="utf-8") as handle:
            json.dump(json_ready(summary), handle, indent=2)
        with (artifact_dir / "diagnostics.json").open("w", encoding="utf-8") as handle:
            json.dump(json_ready(diagnostics), handle, indent=2)

        fold_metrics.to_parquet(artifact_dir / "fold_metrics.parquet")
        equity_curve.to_parquet(artifact_dir / "equity_curve.parquet")
        return artifact_dir

    def run(self, experiment_name: str | None = None) -> ExperimentResult:
        fold_boundaries = self._fold_boundaries()
        if not fold_boundaries:
            raise ValueError("Not enough data to generate any walk-forward folds.")

        experiment_label = experiment_name or getattr(self.strategy, "name", "walk_forward")
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        experiment_id = f"{timestamp}_{experiment_label}"

        fold_frames: list[pd.DataFrame] = []
        fold_metrics: list[dict[str, Any]] = []
        diagnostics: list[dict[str, Any]] = []

        for fold_number, (train_start, test_start, test_end) in enumerate(fold_boundaries, start=1):
            train_data = self.prices.iloc[train_start:test_start].copy()
            test_data = self.prices.iloc[test_start:test_end].copy()

            output = self.strategy.run_fold(train_data=train_data, test_data=test_data)
            evaluated = self._apply_cost_model(output)
            evaluated["fold"] = fold_number
            fold_frames.append(evaluated)

            metrics = self._compute_metrics(evaluated)
            metrics.update(
                {
                    "fold": fold_number,
                    "train_start": train_data.index[0],
                    "train_end": train_data.index[-1],
                    "test_start": test_data.index[0],
                    "test_end": test_data.index[-1],
                }
            )
            fold_metrics.append(metrics)
            diagnostics.append({"fold": fold_number, "diagnostics": output.diagnostics})

        combined = pd.concat(fold_frames, axis=0)
        overall_summary = self._compute_metrics(combined)
        overall_summary.update(
            {
                "experiment_id": experiment_id,
                "folds": len(fold_frames),
                "strategy": experiment_label,
                "cost_model": asdict(self.cost_model),
                "walk_forward_config": asdict(self.config),
            }
        )

        fold_metrics_frame = pd.DataFrame(fold_metrics)
        artifact_dir = self._save_artifacts(
            experiment_id=experiment_id,
            summary=overall_summary,
            fold_metrics=fold_metrics_frame,
            equity_curve=combined,
            diagnostics=diagnostics,
        )

        return ExperimentResult(
            experiment_id=experiment_id,
            summary=overall_summary,
            fold_metrics=fold_metrics_frame,
            equity_curve=combined,
            diagnostics=diagnostics,
            artifact_dir=artifact_dir,
        )


def main() -> None:
    sector_map = {
        "KO": "Beverages",
        "PEP": "Beverages",
        "KDP": "Beverages",
        "XOM": "Energy",
        "CVX": "Energy",
        "COP": "Energy",
        "JPM": "Banks",
        "BAC": "Banks",
        "WFC": "Banks",
        "C": "Banks",
    }

    provider = CachedParquetProvider(
        upstream=YahooFinanceProvider(),
        cache_dir="data/cache",
    )
    prices = provider.get_close_prices(
        symbols=list(sector_map.keys()),
        start="2013-01-01",
        end="2026-04-15",
        interval="1d",
    )

    pipeline = SectorStatArbPipeline(
        sector_map=sector_map,
        portfolio_manager=PortfolioManager(
            max_leverage=1.5,
            risk_per_trade=0.08,
            volatility_window=20,
            max_pair_weight=0.60,
        ),
        screen_config=PairScreenConfig(
            min_history=252,
            correlation_floor=0.60,
            coint_pvalue_threshold=0.10,
            min_half_life=2.0,
            max_half_life=60.0,
            target_half_life=15.0,
        ),
        stat_arb_config=StatArbConfig(
            top_n_pairs=3,
            entry_z=2.0,
            exit_z=0.35,
            break_window=80,
            break_pvalue=0.20,
            transaction_cost_bps=4.0,
        ),
        name="serious_stat_arb",
    )

    backtester = WalkForwardBacktester(
        strategy=pipeline,
        prices=prices,
        config=WalkForwardConfig(
            train_bars=504,
            test_bars=63,
            step_bars=63,
            bars_per_year=252,
        ),
        cost_model=CostModel(
            commission_bps=0.5,
            spread_bps=1.0,
            slippage_bps=0.5,
            borrow_bps_annual=40.0,
        ),
    )

    result = backtester.run(experiment_name="serious_stat_arb")
    print(json.dumps(json_ready(result.summary), indent=2))
    print(f"Artifacts saved to: {result.artifact_dir}")


if __name__ == "__main__":
    main()
