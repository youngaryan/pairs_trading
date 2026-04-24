from __future__ import annotations

import numpy as np
import pandas as pd

from ..framework import StrategyOutput, WalkForwardStrategy


class EventDriftStrategy(WalkForwardStrategy):
    def __init__(
        self,
        symbol: str,
        events: pd.DataFrame,
        holding_period_bars: int = 5,
        entry_threshold: float = 0.15,
        transaction_cost_bps: float = 2.0,
    ) -> None:
        self.symbol = symbol
        self.events = events.copy()
        self.holding_period_bars = holding_period_bars
        self.entry_threshold = entry_threshold
        self.transaction_cost_bps = transaction_cost_bps

    def run_fold(self, train_data: pd.DataFrame, test_data: pd.DataFrame) -> StrategyOutput:
        if self.symbol not in train_data.columns or self.symbol not in test_data.columns:
            raise KeyError(f"{self.symbol} must be present in both train and test data.")

        combined = pd.concat([train_data[[self.symbol]], test_data[[self.symbol]]], axis=0)
        combined = combined[~combined.index.duplicated(keep="last")]
        prices = combined[self.symbol].astype(float)
        analysis = pd.DataFrame(index=prices.index)
        analysis["price"] = prices
        analysis["unit_return"] = prices.pct_change().fillna(0.0)
        analysis["raw_signal"] = 0.0

        symbol_events = self.events.copy()
        if not symbol_events.empty:
            symbol_events["timestamp"] = pd.to_datetime(symbol_events["timestamp"]).dt.tz_localize(None)
            symbol_events["ticker"] = symbol_events["ticker"].astype(str).str.upper()
            symbol_events = symbol_events[symbol_events["ticker"] == self.symbol]
            symbol_events["event_score"] = pd.to_numeric(symbol_events["event_score"], errors="coerce").fillna(0.0)
            symbol_events["confidence"] = pd.to_numeric(symbol_events.get("confidence", 1.0), errors="coerce").fillna(1.0)

            for event in symbol_events.to_dict("records"):
                score = float(event["event_score"])
                if abs(score) < self.entry_threshold:
                    continue

                start_position = analysis.index.searchsorted(pd.Timestamp(event["timestamp"]), side="right")
                if start_position >= len(analysis.index):
                    continue
                end_position = min(len(analysis.index), start_position + self.holding_period_bars)
                signed_score = float(np.clip(score * float(event.get("confidence", 1.0)), -1.0, 1.0))
                analysis.iloc[start_position:end_position, analysis.columns.get_loc("raw_signal")] += signed_score

        analysis["position"] = analysis["raw_signal"].clip(-1.0, 1.0)
        analysis["forecast"] = (analysis["raw_signal"] * 2.0).clip(-2.0, 2.0)
        analysis["gross_return"] = analysis["position"].shift(1).fillna(0.0) * analysis["unit_return"]
        analysis["turnover"] = analysis["position"].diff().abs().fillna(analysis["position"].abs())
        analysis["cost_estimate"] = analysis["turnover"] * (self.transaction_cost_bps / 10_000.0)
        analysis["signal"] = np.sign(analysis["position"]).replace({-0.0: 0.0}).fillna(0.0)
        analysis["short_exposure_per_unit"] = (analysis["position"] < 0.0).astype(float)
        analysis["gross_exposure_per_unit"] = 1.0

        test_frame = analysis.reindex(test_data.index).copy()
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
        ):
            test_frame[column] = test_frame[column].fillna(0.0)

        symbol_events = symbol_events[
            (symbol_events["timestamp"] >= pd.Timestamp(test_data.index[0]))
            & (symbol_events["timestamp"] <= pd.Timestamp(test_data.index[-1]))
        ]
        return StrategyOutput(
            name=f"{self.symbol}_event_drift",
            frame=test_frame,
            diagnostics={
                "symbol": self.symbol,
                "strategy_type": "event_drift",
                "event_count": int(len(symbol_events)),
                "holding_period_bars": float(self.holding_period_bars),
                "entry_threshold": float(self.entry_threshold),
            },
        ).validate(extra_columns=("unit_return", "gross_return"))
