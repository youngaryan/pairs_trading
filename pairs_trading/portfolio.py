from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

from .strategies import StrategyOutput


class PortfolioManager:
    """
    Converts standardized pair-level strategy outputs into a portfolio-level result.

    The allocator uses:
    - rolling spread volatility for risk targeting,
    - forecast magnitude for conviction,
    - estimated trading cost as a penalty term,
    - a portfolio-level leverage cap.
    """

    def __init__(
        self,
        max_leverage: float = 1.5,
        risk_per_trade: float = 0.10,
        volatility_window: int = 20,
        forecast_floor: float = 0.15,
        max_pair_weight: float | None = None,
    ) -> None:
        self.max_leverage = max_leverage
        self.risk_per_trade = risk_per_trade
        self.volatility_window = volatility_window
        self.forecast_floor = forecast_floor
        self.max_pair_weight = max_pair_weight

    def allocate_capital(
        self,
        strategy_outputs: Mapping[str, StrategyOutput],
        portfolio_name: str = "portfolio",
    ) -> StrategyOutput:
        if not strategy_outputs:
            raise ValueError("PortfolioManager needs at least one strategy output.")

        ordered_outputs = dict(strategy_outputs)
        for name, output in ordered_outputs.items():
            output.validate(extra_columns=("spread_return",))
            if output.frame.empty:
                raise ValueError(f"Strategy output '{name}' is empty.")

        portfolio_index = ordered_outputs[next(iter(ordered_outputs))].frame.index
        for output in ordered_outputs.values():
            portfolio_index = portfolio_index.union(output.frame.index)
        portfolio_index = portfolio_index.sort_values()

        weights = pd.DataFrame(index=portfolio_index)
        pair_returns = pd.DataFrame(index=portfolio_index)
        pair_costs = pd.DataFrame(index=portfolio_index)
        pair_short_exposure = pd.DataFrame(index=portfolio_index)
        pair_gross_exposure = pd.DataFrame(index=portfolio_index)
        pair_forecasts = pd.DataFrame(index=portfolio_index)
        pair_sentiment_strength = pd.DataFrame(index=portfolio_index)
        pair_sentiment_confidence = pd.DataFrame(index=portfolio_index)

        pair_summaries: list[dict[str, float | str]] = []

        for pair_name, output in ordered_outputs.items():
            frame = output.frame.reindex(portfolio_index).copy()
            for column in ("signal", "forecast", "position", "cost_estimate", "spread_return"):
                frame[column] = frame[column].fillna(0.0)
            frame["short_exposure_per_unit"] = frame.get("short_exposure_per_unit", 0.5)
            frame["gross_exposure_per_unit"] = frame.get("gross_exposure_per_unit", 1.0)
            frame["sentiment_strength"] = frame.get("sentiment_strength", 0.0)
            frame["sentiment_confidence"] = frame.get("sentiment_confidence", 0.0)

            rolling_vol = frame["spread_return"].rolling(self.volatility_window).std()
            fallback_vol = frame["spread_return"].std()
            fallback_vol = float(fallback_vol) if pd.notna(fallback_vol) and fallback_vol > 0 else 0.01
            rolling_vol = rolling_vol.replace(0.0, np.nan).fillna(fallback_vol)

            conviction = frame["forecast"].abs().clip(lower=self.forecast_floor, upper=2.0)
            cost_penalty = (1.0 / (1.0 + frame["cost_estimate"] * 10_000.0)).clip(lower=0.25, upper=1.0)
            gross_weight = (self.risk_per_trade / rolling_vol) * conviction * cost_penalty
            if self.max_pair_weight is not None:
                gross_weight = gross_weight.clip(upper=self.max_pair_weight)

            signed_weight = gross_weight * frame["position"]

            weights[pair_name] = signed_weight.fillna(0.0)
            pair_returns[pair_name] = frame["spread_return"]
            pair_costs[pair_name] = frame["cost_estimate"]
            pair_short_exposure[pair_name] = frame["short_exposure_per_unit"]
            pair_gross_exposure[pair_name] = frame["gross_exposure_per_unit"]
            pair_forecasts[pair_name] = frame["forecast"]
            pair_sentiment_strength[pair_name] = frame["sentiment_strength"]
            pair_sentiment_confidence[pair_name] = frame["sentiment_confidence"]

            pair_summaries.append(
                {
                    "pair": pair_name,
                    "mean_abs_forecast": float(frame["forecast"].abs().mean()),
                    "mean_cost_estimate": float(frame["cost_estimate"].mean()),
                    "fallback_vol": fallback_vol,
                    **output.diagnostics,
                }
            )

        gross_leverage = weights.abs().sum(axis=1)
        scaling = pd.Series(1.0, index=portfolio_index)
        over_limit = gross_leverage > self.max_leverage
        scaling.loc[over_limit] = self.max_leverage / gross_leverage.loc[over_limit]

        scaled_weights = weights.mul(scaling, axis=0)
        lagged_weights = scaled_weights.shift(1).fillna(0.0)

        gross_return = (lagged_weights * pair_returns).sum(axis=1)
        cost_estimate = (scaled_weights.abs() * pair_costs).sum(axis=1)
        short_exposure = (lagged_weights.abs() * pair_short_exposure).sum(axis=1)
        deployed_gross = (lagged_weights.abs() * pair_gross_exposure).sum(axis=1)
        turnover = scaled_weights.diff().abs().fillna(scaled_weights.abs()).sum(axis=1)

        weight_abs_sum = scaled_weights.abs().sum(axis=1).replace(0.0, np.nan)
        forecast = (scaled_weights.abs() * pair_forecasts.abs()).sum(axis=1) / weight_abs_sum
        forecast = forecast.fillna(0.0)
        portfolio_sentiment_strength = (scaled_weights.abs() * pair_sentiment_strength).sum(axis=1) / weight_abs_sum
        portfolio_sentiment_confidence = (scaled_weights.abs() * pair_sentiment_confidence).sum(axis=1) / weight_abs_sum
        portfolio_sentiment_strength = portfolio_sentiment_strength.fillna(0.0)
        portfolio_sentiment_confidence = portfolio_sentiment_confidence.fillna(0.0)

        portfolio_frame = pd.DataFrame(index=portfolio_index)
        portfolio_frame["signal"] = np.sign(scaled_weights.sum(axis=1)).fillna(0.0)
        portfolio_frame["forecast"] = forecast
        portfolio_frame["position"] = scaled_weights.abs().sum(axis=1)
        portfolio_frame["cost_estimate"] = cost_estimate
        portfolio_frame["gross_return"] = gross_return
        portfolio_frame["turnover"] = turnover
        portfolio_frame["short_exposure"] = short_exposure
        portfolio_frame["gross_exposure"] = deployed_gross
        portfolio_frame["sentiment_strength"] = portfolio_sentiment_strength
        portfolio_frame["sentiment_confidence"] = portfolio_sentiment_confidence

        for pair_name in scaled_weights.columns:
            portfolio_frame[f"weight_{pair_name}"] = scaled_weights[pair_name]

        diagnostics = {
            "portfolio_name": portfolio_name,
            "pair_count": len(strategy_outputs),
            "pair_summaries": pair_summaries,
        }

        return StrategyOutput(
            name=portfolio_name,
            frame=portfolio_frame,
            diagnostics=diagnostics,
        ).validate(extra_columns=("gross_return",))
