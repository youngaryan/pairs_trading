from __future__ import annotations

import unittest

from pairs_trading.backtesting import CostModel, WalkForwardBacktester, WalkForwardConfig
from pairs_trading.pipelines import DirectionalPipelineConfig, DirectionalStrategyPipeline
from pairs_trading.portfolio import PortfolioManager
from pairs_trading.strategies import DonchianBreakoutStrategy, MovingAverageCrossStrategy, RSIMeanReversionStrategy
from tests.common import fresh_test_dir, synthetic_directional_prices


class DirectionalStrategyTests(unittest.TestCase):
    def test_moving_average_cross_emits_standardized_output(self) -> None:
        prices = synthetic_directional_prices()
        strategy = MovingAverageCrossStrategy(symbol="TREND", fast_window=15, slow_window=60)
        output = strategy.run_fold(train_data=prices.iloc[:300][["TREND"]], test_data=prices.iloc[300:380][["TREND"]])

        self.assertIn("unit_return", output.frame.columns)
        self.assertIn("gross_return", output.frame.columns)
        self.assertGreater(float(output.frame["forecast"].abs().sum()), 0.0)
        self.assertGreater(float(output.frame["position"].abs().sum()), 0.0)

    def test_rsi_mean_reversion_and_donchian_breakout_take_positions(self) -> None:
        prices = synthetic_directional_prices()
        rsi_strategy = RSIMeanReversionStrategy(symbol="MEAN", rsi_window=10, lower_entry=35.0, upper_entry=65.0)
        breakout_strategy = DonchianBreakoutStrategy(symbol="BREAK", breakout_window=40, exit_window=15)

        rsi_output = rsi_strategy.run_fold(train_data=prices.iloc[:300][["MEAN"]], test_data=prices.iloc[300:380][["MEAN"]])
        breakout_output = breakout_strategy.run_fold(train_data=prices.iloc[:320][["BREAK"]], test_data=prices.iloc[320:420][["BREAK"]])

        self.assertGreater(float(rsi_output.frame["position"].abs().sum()), 0.0)
        self.assertGreater(float(breakout_output.frame["position"].abs().sum()), 0.0)


class DirectionalPipelineTests(unittest.TestCase):
    def test_directional_pipeline_runs_walk_forward_backtest(self) -> None:
        prices = synthetic_directional_prices()
        pipeline = DirectionalStrategyPipeline(
            strategy_factory=lambda symbol: MovingAverageCrossStrategy(symbol=symbol, fast_window=15, slow_window=60),
            portfolio_manager=PortfolioManager(
                max_leverage=1.0,
                risk_per_trade=0.03,
                volatility_window=15,
                max_strategy_weight=0.4,
            ),
            config=DirectionalPipelineConfig.from_symbols(["TREND", "MEAN", "BREAK"], min_history=120),
            name="directional_smoke",
        )

        result = WalkForwardBacktester(
            strategy=pipeline,
            prices=prices,
            config=WalkForwardConfig(train_bars=260, test_bars=80, step_bars=80, bars_per_year=252),
            cost_model=CostModel(commission_bps=0.2, spread_bps=0.5, slippage_bps=0.3, borrow_bps_annual=15.0),
            experiment_root=fresh_test_dir("artifacts/test_runs/directional"),
        ).run("directional_smoke")

        self.assertGreater(len(result.fold_metrics), 0)
        self.assertTrue(any(column.startswith("weight_") for column in result.equity_curve.columns))
        self.assertTrue(result.artifact_dir.exists())


if __name__ == "__main__":
    unittest.main()
