from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..framework import StrategyOutput, WalkForwardStrategy


def _extract_price_series(train_data: pd.DataFrame, test_data: pd.DataFrame, symbol: str | None = None) -> tuple[str, pd.Series, pd.Index]:
    train_frame = train_data.copy()
    test_frame = test_data.copy()
    if symbol is None:
        if len(train_frame.columns) != 1 or len(test_frame.columns) != 1:
            raise ValueError("Single-asset strategies require exactly one column unless a symbol is provided.")
        symbol = str(train_frame.columns[0])

    if symbol not in train_frame.columns or symbol not in test_frame.columns:
        raise KeyError(f"Symbol '{symbol}' must exist in both train and test data.")

    combined = pd.concat([train_frame[[symbol]], test_frame[[symbol]]], axis=0)
    combined = combined[~combined.index.duplicated(keep="last")]
    return symbol, combined[symbol].astype(float), test_frame.index


def _build_standard_output(
    *,
    name: str,
    analysis: pd.DataFrame,
    test_index: pd.Index,
    transaction_cost_bps: float,
    diagnostics: dict[str, float | str],
) -> StrategyOutput:
    frame = analysis.reindex(test_index).copy()
    frame["turnover"] = frame["position"].diff().abs().fillna(frame["position"].abs())
    frame["cost_estimate"] = frame["turnover"] * (transaction_cost_bps / 10_000.0)
    frame["gross_return"] = frame["position"].shift(1).fillna(0.0) * frame["unit_return"].fillna(0.0)
    frame["signal"] = np.sign(frame["position"]).replace({-0.0: 0.0}).fillna(0.0)
    frame["short_exposure_per_unit"] = (frame["position"] < 0.0).astype(float)
    frame["gross_exposure_per_unit"] = 1.0
    frame["instrument_return"] = frame["unit_return"].fillna(0.0)

    for column in (
        "signal",
        "forecast",
        "position",
        "cost_estimate",
        "unit_return",
        "gross_return",
        "turnover",
        "short_exposure_per_unit",
        "gross_exposure_per_unit",
        "instrument_return",
    ):
        frame[column] = frame[column].fillna(0.0)

    return StrategyOutput(name=name, frame=frame, diagnostics=diagnostics).validate(extra_columns=("unit_return", "gross_return"))


class MovingAverageCrossStrategy(WalkForwardStrategy):
    def __init__(
        self,
        symbol: str,
        fast_window: int = 20,
        slow_window: int = 80,
        transaction_cost_bps: float = 2.0,
    ) -> None:
        if fast_window >= slow_window:
            raise ValueError("fast_window must be smaller than slow_window.")
        self.symbol = symbol
        self.fast_window = fast_window
        self.slow_window = slow_window
        self.transaction_cost_bps = transaction_cost_bps

    def run_fold(self, train_data: pd.DataFrame, test_data: pd.DataFrame) -> StrategyOutput:
        symbol, prices, test_index = _extract_price_series(train_data, test_data, self.symbol)
        analysis = pd.DataFrame(index=prices.index)
        analysis["price"] = prices
        analysis["unit_return"] = prices.pct_change().fillna(0.0)
        analysis["fast_ma"] = prices.rolling(self.fast_window).mean()
        analysis["slow_ma"] = prices.rolling(self.slow_window).mean()
        spread = analysis["fast_ma"] / analysis["slow_ma"] - 1.0
        analysis["forecast"] = (spread * 100.0).clip(-2.0, 2.0).fillna(0.0)
        analysis["position"] = np.sign(spread).replace({-0.0: 0.0}).fillna(0.0)
        analysis.loc[analysis["slow_ma"].isna(), "position"] = 0.0

        return _build_standard_output(
            name=f"{symbol}_ma_cross",
            analysis=analysis,
            test_index=test_index,
            transaction_cost_bps=self.transaction_cost_bps,
            diagnostics={
                "symbol": symbol,
                "strategy_type": "moving_average_cross",
                "fast_window": float(self.fast_window),
                "slow_window": float(self.slow_window),
            },
        )


class RSIMeanReversionStrategy(WalkForwardStrategy):
    def __init__(
        self,
        symbol: str,
        rsi_window: int = 14,
        lower_entry: float = 30.0,
        upper_entry: float = 70.0,
        exit_level: float = 50.0,
        transaction_cost_bps: float = 2.0,
    ) -> None:
        self.symbol = symbol
        self.rsi_window = rsi_window
        self.lower_entry = lower_entry
        self.upper_entry = upper_entry
        self.exit_level = exit_level
        self.transaction_cost_bps = transaction_cost_bps

    def _compute_rsi(self, prices: pd.Series) -> pd.Series:
        delta = prices.diff()
        gains = delta.clip(lower=0.0)
        losses = -delta.clip(upper=0.0)
        avg_gain = gains.ewm(alpha=1 / self.rsi_window, adjust=False, min_periods=self.rsi_window).mean()
        avg_loss = losses.ewm(alpha=1 / self.rsi_window, adjust=False, min_periods=self.rsi_window).mean()
        relative_strength = avg_gain / avg_loss.replace(0.0, np.nan)
        rsi = 100.0 - (100.0 / (1.0 + relative_strength))
        return rsi.fillna(50.0)

    def run_fold(self, train_data: pd.DataFrame, test_data: pd.DataFrame) -> StrategyOutput:
        symbol, prices, test_index = _extract_price_series(train_data, test_data, self.symbol)
        analysis = pd.DataFrame(index=prices.index)
        analysis["price"] = prices
        analysis["unit_return"] = prices.pct_change().fillna(0.0)
        analysis["rsi"] = self._compute_rsi(prices)
        analysis["forecast"] = ((50.0 - analysis["rsi"]) / 20.0).clip(-2.0, 2.0).fillna(0.0)

        positions = np.zeros(len(analysis))
        current_pos = 0.0
        for index in range(len(analysis)):
            rsi = float(analysis["rsi"].iloc[index])
            if np.isnan(rsi):
                positions[index] = 0.0
                continue
            if current_pos == 0.0:
                if rsi <= self.lower_entry:
                    current_pos = 1.0
                elif rsi >= self.upper_entry:
                    current_pos = -1.0
            elif current_pos > 0.0 and rsi >= self.exit_level:
                current_pos = 0.0
            elif current_pos < 0.0 and rsi <= self.exit_level:
                current_pos = 0.0
            positions[index] = current_pos

        analysis["position"] = positions

        return _build_standard_output(
            name=f"{symbol}_rsi_mean_reversion",
            analysis=analysis,
            test_index=test_index,
            transaction_cost_bps=self.transaction_cost_bps,
            diagnostics={
                "symbol": symbol,
                "strategy_type": "rsi_mean_reversion",
                "rsi_window": float(self.rsi_window),
                "lower_entry": float(self.lower_entry),
                "upper_entry": float(self.upper_entry),
            },
        )


class DonchianBreakoutStrategy(WalkForwardStrategy):
    def __init__(
        self,
        symbol: str,
        breakout_window: int = 55,
        exit_window: int = 20,
        transaction_cost_bps: float = 2.0,
    ) -> None:
        if exit_window >= breakout_window:
            raise ValueError("exit_window must be smaller than breakout_window.")
        self.symbol = symbol
        self.breakout_window = breakout_window
        self.exit_window = exit_window
        self.transaction_cost_bps = transaction_cost_bps

    def run_fold(self, train_data: pd.DataFrame, test_data: pd.DataFrame) -> StrategyOutput:
        symbol, prices, test_index = _extract_price_series(train_data, test_data, self.symbol)
        analysis = pd.DataFrame(index=prices.index)
        analysis["price"] = prices
        analysis["unit_return"] = prices.pct_change().fillna(0.0)
        analysis["upper_band"] = prices.shift(1).rolling(self.breakout_window).max()
        analysis["lower_band"] = prices.shift(1).rolling(self.breakout_window).min()
        analysis["exit_high"] = prices.shift(1).rolling(self.exit_window).max()
        analysis["exit_low"] = prices.shift(1).rolling(self.exit_window).min()

        positions = np.zeros(len(analysis))
        forecasts = np.zeros(len(analysis))
        current_pos = 0.0
        for index in range(len(analysis)):
            price = float(analysis["price"].iloc[index])
            upper = analysis["upper_band"].iloc[index]
            lower = analysis["lower_band"].iloc[index]
            exit_high = analysis["exit_high"].iloc[index]
            exit_low = analysis["exit_low"].iloc[index]

            if pd.isna(upper) or pd.isna(lower):
                positions[index] = 0.0
                continue

            if current_pos == 0.0:
                if price > float(upper):
                    current_pos = 1.0
                elif price < float(lower):
                    current_pos = -1.0
            elif current_pos > 0.0 and price < float(exit_low):
                current_pos = 0.0
            elif current_pos < 0.0 and price > float(exit_high):
                current_pos = 0.0

            positions[index] = current_pos
            channel_width = max(float(upper) - float(lower), 1e-8)
            forecasts[index] = float(np.clip((price - (float(upper) + float(lower)) / 2.0) / channel_width * 4.0, -2.0, 2.0))

        analysis["position"] = positions
        analysis["forecast"] = forecasts

        return _build_standard_output(
            name=f"{symbol}_donchian_breakout",
            analysis=analysis,
            test_index=test_index,
            transaction_cost_bps=self.transaction_cost_bps,
            diagnostics={
                "symbol": symbol,
                "strategy_type": "donchian_breakout",
                "breakout_window": float(self.breakout_window),
                "exit_window": float(self.exit_window),
            },
        )
