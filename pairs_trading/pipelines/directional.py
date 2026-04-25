from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import pandas as pd

from ..core.framework import StrategyOutput, WalkForwardStrategy
from ..core.portfolio import PortfolioManager


StrategyFactory = Callable[[str], WalkForwardStrategy]


@dataclass(frozen=True)
class DirectionalPipelineConfig:
    min_history: int = 120
    symbols: tuple[str, ...] | None = None

    @classmethod
    def from_symbols(cls, symbols: Sequence[str] | None, min_history: int = 120) -> "DirectionalPipelineConfig":
        return cls(
            min_history=min_history,
            symbols=None if symbols is None else tuple(dict.fromkeys(symbols)),
        )


class DirectionalStrategyPipeline(WalkForwardStrategy):
    def __init__(
        self,
        strategy_factory: StrategyFactory,
        portfolio_manager: PortfolioManager | None = None,
        config: DirectionalPipelineConfig = DirectionalPipelineConfig(),
        name: str = "directional_pipeline",
    ) -> None:
        self.strategy_factory = strategy_factory
        self.portfolio_manager = portfolio_manager or PortfolioManager()
        self.config = config
        self.name = name

    def _flat_output(self, index: pd.Index, reason: str) -> StrategyOutput:
        frame = pd.DataFrame(index=index)
        frame["signal"] = 0.0
        frame["forecast"] = 0.0
        frame["position"] = 0.0
        frame["cost_estimate"] = 0.0
        frame["gross_return"] = 0.0
        frame["unit_return"] = 0.0
        frame["turnover"] = 0.0
        frame["short_exposure"] = 0.0
        frame["gross_exposure"] = 0.0
        return StrategyOutput(
            name=self.name,
            frame=frame,
            diagnostics={"status": reason, "selected_symbols": []},
        ).validate(extra_columns=("unit_return", "gross_return"))

    def run_fold(self, train_data: pd.DataFrame, test_data: pd.DataFrame) -> StrategyOutput:
        candidate_symbols = list(self.config.symbols or tuple(train_data.columns))
        available_symbols = [symbol for symbol in candidate_symbols if symbol in train_data.columns and symbol in test_data.columns]
        available_symbols = [
            symbol
            for symbol in available_symbols
            if train_data[symbol].dropna().shape[0] >= self.config.min_history and test_data[symbol].dropna().shape[0] > 0
        ]

        if not available_symbols:
            return self._flat_output(index=test_data.index, reason="no_available_symbols")

        outputs: dict[str, StrategyOutput] = {}
        for symbol in available_symbols:
            strategy = self.strategy_factory(symbol)
            outputs[symbol] = strategy.run_fold(
                train_data=train_data[[symbol]],
                test_data=test_data[[symbol]],
            )

        portfolio_output = self.portfolio_manager.allocate_capital(
            strategy_outputs=outputs,
            portfolio_name=self.name,
        )
        portfolio_output.diagnostics.update(
            {
                "selected_symbols": available_symbols,
                "strategy_count": len(outputs),
                "pipeline_type": "directional",
            }
        )
        return portfolio_output
