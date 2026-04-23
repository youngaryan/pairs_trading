from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import pandas as pd

from ..framework import StrategyOutput, WalkForwardStrategy
from ..portfolio import PortfolioManager
from ..research import PairScreenConfig, rank_sector_pairs
from ..sentiment import (
    SentimentConfig,
    adjust_pair_rankings_with_sentiment,
    apply_sentiment_overlay,
    build_pair_sentiment_overlay,
)
from ..strategies import KalmanPairsStrategy


@dataclass(frozen=True)
class StatArbConfig:
    top_n_pairs: int = 3
    entry_z: float = 2.0
    exit_z: float = 0.35
    break_window: int = 80
    break_pvalue: float = 0.20
    transaction_cost_bps: float = 4.0


def select_diversified_pairs(ranked_pairs: pd.DataFrame, top_n_pairs: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    used_tickers: set[str] = set()

    for record in ranked_pairs.to_dict("records"):
        ticker1 = record["Ticker_1"]
        ticker2 = record["Ticker_2"]
        if ticker1 in used_tickers or ticker2 in used_tickers:
            continue
        selected.append(record)
        used_tickers.update({ticker1, ticker2})
        if len(selected) >= top_n_pairs:
            return selected

    if len(selected) >= top_n_pairs:
        return selected

    for record in ranked_pairs.to_dict("records"):
        if record in selected:
            continue
        selected.append(record)
        if len(selected) >= top_n_pairs:
            break

    return selected


class SectorStatArbPipeline(WalkForwardStrategy):
    def __init__(
        self,
        sector_map: Mapping[str, str],
        portfolio_manager: PortfolioManager | None = None,
        screen_config: PairScreenConfig = PairScreenConfig(),
        stat_arb_config: StatArbConfig = StatArbConfig(),
        daily_sentiment: pd.DataFrame | None = None,
        sentiment_config: SentimentConfig | None = None,
        name: str = "sector_stat_arb",
    ) -> None:
        self.sector_map = dict(sector_map)
        self.portfolio_manager = portfolio_manager or PortfolioManager()
        self.screen_config = screen_config
        self.stat_arb_config = stat_arb_config
        self.daily_sentiment = daily_sentiment
        self.sentiment_config = sentiment_config
        self.name = name

    def _flat_output(self, index: pd.Index, reason: str) -> StrategyOutput:
        frame = pd.DataFrame(index=index)
        frame["signal"] = 0.0
        frame["forecast"] = 0.0
        frame["position"] = 0.0
        frame["cost_estimate"] = 0.0
        frame["unit_return"] = 0.0
        frame["gross_return"] = 0.0
        frame["turnover"] = 0.0
        frame["short_exposure"] = 0.0
        frame["gross_exposure"] = 0.0
        return StrategyOutput(
            name=self.name,
            frame=frame,
            diagnostics={"status": reason, "selected_pairs": []},
        ).validate(extra_columns=("unit_return", "gross_return"))

    def run_fold(self, train_data: pd.DataFrame, test_data: pd.DataFrame) -> StrategyOutput:
        ranked_pairs = rank_sector_pairs(
            prices=train_data,
            sector_map=self.sector_map,
            screen_config=self.screen_config,
        )
        if self.daily_sentiment is not None and self.sentiment_config is not None and not ranked_pairs.empty:
            ranked_pairs = adjust_pair_rankings_with_sentiment(
                ranked_pairs=ranked_pairs,
                daily_sentiment=self.daily_sentiment,
                asof_date=train_data.index[-1],
                config=self.sentiment_config,
            )
        if ranked_pairs.empty:
            return self._flat_output(index=test_data.index, reason="no_ranked_pairs")

        selected_pairs = select_diversified_pairs(
            ranked_pairs=ranked_pairs,
            top_n_pairs=self.stat_arb_config.top_n_pairs,
        )
        if not selected_pairs:
            return self._flat_output(index=test_data.index, reason="no_selected_pairs")

        pair_outputs: dict[str, StrategyOutput] = {}
        for pair_record in selected_pairs:
            ticker1 = pair_record["Ticker_1"]
            ticker2 = pair_record["Ticker_2"]
            strategy = KalmanPairsStrategy(
                ticker1=ticker1,
                ticker2=ticker2,
                entry_z=self.stat_arb_config.entry_z,
                exit_z=self.stat_arb_config.exit_z,
                break_window=self.stat_arb_config.break_window,
                break_pvalue=self.stat_arb_config.break_pvalue,
                transaction_cost_bps=self.stat_arb_config.transaction_cost_bps,
                pair_metadata=pair_record,
            )
            pair_name = f"{ticker1}_{ticker2}"
            pair_outputs[pair_name] = strategy.run_fold(
                train_data=train_data[[ticker1, ticker2]],
                test_data=test_data[[ticker1, ticker2]],
            )
            if self.daily_sentiment is not None and self.sentiment_config is not None:
                overlay = build_pair_sentiment_overlay(
                    daily_sentiment=self.daily_sentiment,
                    ticker1=ticker1,
                    ticker2=ticker2,
                    index=test_data.index,
                    config=self.sentiment_config,
                )
                pair_outputs[pair_name] = apply_sentiment_overlay(
                    strategy_output=pair_outputs[pair_name],
                    sentiment_overlay=overlay,
                    config=self.sentiment_config,
                )

        portfolio_output = self.portfolio_manager.allocate_capital(
            strategy_outputs=pair_outputs,
            portfolio_name=self.name,
        )
        portfolio_output.diagnostics.update(
            {
                "selected_pairs": selected_pairs,
                "candidate_count": int(len(ranked_pairs)),
                "screen_config": self.screen_config.__dict__,
                "stat_arb_config": self.stat_arb_config.__dict__,
                "sentiment_enabled": self.daily_sentiment is not None and self.sentiment_config is not None,
            }
        )
        return portfolio_output
