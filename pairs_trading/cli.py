from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Callable

from .backtesting import CostModel, ExperimentResult, WalkForwardBacktester, WalkForwardConfig, json_ready
from .market_data import CachedParquetProvider, YahooFinanceProvider
from .news_data import (
    AlphaVantageNewsProvider,
    BenzingaNewsProvider,
    CachedNewsSentimentProvider,
    CompositeHeadlineProvider,
    DailySentimentFileProvider,
    LocalNewsFileProvider,
)
from .pipelines import DirectionalPipelineConfig, DirectionalStrategyPipeline, SectorStatArbPipeline, StatArbConfig
from .portfolio import PortfolioManager
from .research import PairScreenConfig
from .sentiment import FinBERTSentimentModel, SentimentConfig, build_best_available_sentiment_model
from .strategies import DonchianBreakoutStrategy, MovingAverageCrossStrategy, RSIMeanReversionStrategy
from .visualization import ExperimentVisualizer


DEFAULT_SECTOR_MAP = {
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


def load_sector_map(path: str | Path | None) -> dict[str, str]:
    if path is None:
        return dict(DEFAULT_SECTOR_MAP)

    source = Path(path)
    if source.suffix.lower() == ".json":
        data = json.loads(source.read_text(encoding="utf-8"))
        return {str(ticker).upper(): str(sector) for ticker, sector in data.items()}

    if source.suffix.lower() == ".csv":
        import pandas as pd

        frame = pd.read_csv(source)
        if not {"ticker", "sector"} <= set(frame.columns):
            raise ValueError("Sector CSV must contain 'ticker' and 'sector' columns.")
        return {
            str(row["ticker"]).upper(): str(row["sector"])
            for _, row in frame[["ticker", "sector"]].dropna().iterrows()
        }

    raise ValueError(f"Unsupported sector map format: {source.suffix}")


def load_daily_sentiment(
    tickers: list[str],
    start: str,
    end: str,
    news_provider_names: list[str] | None,
    news_files: list[str] | None,
    daily_sentiment_file: str | None,
    use_finbert: bool,
    local_finbert_only: bool,
    sentiment_cache_dir: str,
    news_api_key: str | None,
    alphavantage_api_key: str | None,
    benzinga_api_key: str | None,
    news_topics: list[str] | None = None,
):
    if daily_sentiment_file:
        provider = DailySentimentFileProvider(daily_sentiment_file)
        return provider.get_daily_sentiment(tickers=tickers, start=start, end=end)

    if not news_provider_names:
        return None

    providers = []
    provider_names = list(dict.fromkeys(news_provider_names))

    if "local" in provider_names:
        if not news_files:
            raise ValueError("The local news provider requires at least one --news-file.")
        providers.extend(LocalNewsFileProvider(news_file) for news_file in news_files)

    if "alphavantage" in provider_names:
        api_key = alphavantage_api_key or news_api_key or os.getenv("ALPHAVANTAGE_API_KEY")
        if not api_key:
            raise ValueError("Alpha Vantage news requires --alphavantage-api-key, --news-api-key, or ALPHAVANTAGE_API_KEY.")
        providers.append(AlphaVantageNewsProvider(api_key=api_key, topics=news_topics))

    if "benzinga" in provider_names:
        api_key = benzinga_api_key or news_api_key or os.getenv("BENZINGA_API_KEY")
        if not api_key:
            raise ValueError("Benzinga news requires --benzinga-api-key, --news-api-key, or BENZINGA_API_KEY.")
        providers.append(BenzingaNewsProvider(api_key=api_key))

    if not providers:
        return None

    model = FinBERTSentimentModel(local_files_only=local_finbert_only) if use_finbert else build_best_available_sentiment_model()
    headline_provider = providers[0] if len(providers) == 1 else CompositeHeadlineProvider(providers)
    provider = CachedNewsSentimentProvider(
        headline_provider=headline_provider,
        sentiment_model=model,
        cache_dir=sentiment_cache_dir,
    )
    return provider.get_daily_sentiment(tickers=tickers, start=start, end=end)


def _build_directional_strategy_factory(
    strategy_name: str,
    *,
    fast_window: int,
    slow_window: int,
    rsi_window: int,
    lower_entry: float,
    upper_entry: float,
    exit_level: float,
    breakout_window: int,
    breakout_exit_window: int,
    strategy_cost_bps: float,
) -> tuple[Callable[[str], object], int]:
    if strategy_name == "ma_cross":
        return (
            lambda symbol: MovingAverageCrossStrategy(
                symbol=symbol,
                fast_window=fast_window,
                slow_window=slow_window,
                transaction_cost_bps=strategy_cost_bps,
            ),
            max(120, slow_window + 20),
        )
    if strategy_name == "rsi_mean_reversion":
        return (
            lambda symbol: RSIMeanReversionStrategy(
                symbol=symbol,
                rsi_window=rsi_window,
                lower_entry=lower_entry,
                upper_entry=upper_entry,
                exit_level=exit_level,
                transaction_cost_bps=strategy_cost_bps,
            ),
            max(80, rsi_window * 4),
        )
    if strategy_name == "donchian_breakout":
        return (
            lambda symbol: DonchianBreakoutStrategy(
                symbol=symbol,
                breakout_window=breakout_window,
                exit_window=breakout_exit_window,
                transaction_cost_bps=strategy_cost_bps,
            ),
            max(120, breakout_window + breakout_exit_window + 20),
        )
    raise ValueError(f"Unsupported directional strategy: {strategy_name}")


def run_stat_arb_pipeline(
    sector_map_path: str | None = None,
    start: str = "2018-01-01",
    end: str = "2026-04-15",
    interval: str = "1d",
    experiment_name: str = "serious_stat_arb",
    price_cache_dir: str = "data/cache",
    sentiment_cache_dir: str = "data/sentiment_cache",
    artifact_root: str = "artifacts/experiments",
    news_provider_names: list[str] | None = None,
    news_files: list[str] | None = None,
    daily_sentiment_file: str | None = None,
    use_finbert: bool = False,
    local_finbert_only: bool = False,
    news_api_key: str | None = None,
    alphavantage_api_key: str | None = None,
    benzinga_api_key: str | None = None,
    news_topics: list[str] | None = None,
) -> dict[str, Any]:
    sector_map = load_sector_map(sector_map_path)
    tickers = list(sector_map.keys())

    price_provider = CachedParquetProvider(
        upstream=YahooFinanceProvider(),
        cache_dir=price_cache_dir,
    )
    prices = price_provider.get_close_prices(
        symbols=tickers,
        start=start,
        end=end,
        interval=interval,
    )

    daily_sentiment = load_daily_sentiment(
        tickers=tickers,
        start=start,
        end=end,
        news_provider_names=news_provider_names,
        news_files=news_files,
        daily_sentiment_file=daily_sentiment_file,
        use_finbert=use_finbert,
        local_finbert_only=local_finbert_only,
        sentiment_cache_dir=sentiment_cache_dir,
        news_api_key=news_api_key,
        alphavantage_api_key=alphavantage_api_key,
        benzinga_api_key=benzinga_api_key,
        news_topics=news_topics,
    )

    pipeline = SectorStatArbPipeline(
        sector_map=sector_map,
        portfolio_manager=PortfolioManager(
            max_leverage=1.5,
            risk_per_trade=0.08,
            volatility_window=20,
            max_strategy_weight=0.60,
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
        daily_sentiment=daily_sentiment,
        sentiment_config=SentimentConfig() if daily_sentiment is not None else None,
        name=experiment_name,
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
        experiment_root=artifact_root,
    )

    result = backtester.run(experiment_name=experiment_name)
    visuals = ExperimentVisualizer(result.artifact_dir / "visuals").create_dashboard(result)
    return {
        "result": result,
        "visuals": visuals,
        "summary": json_ready(result.summary),
    }


def run_directional_pipeline(
    strategy_name: str,
    symbols: list[str],
    start: str = "2018-01-01",
    end: str = "2026-04-15",
    interval: str = "1d",
    experiment_name: str | None = None,
    price_cache_dir: str = "data/cache",
    artifact_root: str = "artifacts/experiments",
    train_bars: int = 252,
    test_bars: int = 63,
    step_bars: int = 63,
    bars_per_year: int = 252,
    fast_window: int = 20,
    slow_window: int = 80,
    rsi_window: int = 14,
    lower_entry: float = 30.0,
    upper_entry: float = 70.0,
    exit_level: float = 50.0,
    breakout_window: int = 55,
    breakout_exit_window: int = 20,
    strategy_cost_bps: float = 2.0,
) -> dict[str, Any]:
    if not symbols:
        raise ValueError("Directional pipelines require at least one symbol.")

    strategy_factory, min_history = _build_directional_strategy_factory(
        strategy_name,
        fast_window=fast_window,
        slow_window=slow_window,
        rsi_window=rsi_window,
        lower_entry=lower_entry,
        upper_entry=upper_entry,
        exit_level=exit_level,
        breakout_window=breakout_window,
        breakout_exit_window=breakout_exit_window,
        strategy_cost_bps=strategy_cost_bps,
    )

    price_provider = CachedParquetProvider(
        upstream=YahooFinanceProvider(),
        cache_dir=price_cache_dir,
    )
    prices = price_provider.get_close_prices(
        symbols=symbols,
        start=start,
        end=end,
        interval=interval,
    )

    pipeline_name = experiment_name or strategy_name
    pipeline = DirectionalStrategyPipeline(
        strategy_factory=strategy_factory,
        portfolio_manager=PortfolioManager(
            max_leverage=1.25,
            risk_per_trade=0.06,
            volatility_window=20,
            max_strategy_weight=0.35,
        ),
        config=DirectionalPipelineConfig.from_symbols(symbols=symbols, min_history=min_history),
        name=pipeline_name,
    )

    backtester = WalkForwardBacktester(
        strategy=pipeline,
        prices=prices,
        config=WalkForwardConfig(
            train_bars=train_bars,
            test_bars=test_bars,
            step_bars=step_bars,
            bars_per_year=bars_per_year,
        ),
        cost_model=CostModel(
            commission_bps=0.5,
            spread_bps=1.0,
            slippage_bps=0.5,
            borrow_bps_annual=25.0,
        ),
        experiment_root=artifact_root,
    )

    result = backtester.run(experiment_name=pipeline_name)
    visuals = ExperimentVisualizer(result.artifact_dir / "visuals").create_dashboard(result)
    return {
        "result": result,
        "visuals": visuals,
        "summary": json_ready(result.summary),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the quant walk-forward research pipeline.")
    parser.add_argument(
        "--pipeline",
        default="stat_arb",
        choices=["stat_arb", "ma_cross", "rsi_mean_reversion", "donchian_breakout"],
        help="Research pipeline to run.",
    )
    parser.add_argument("--symbols", nargs="*", help="Symbols for directional pipelines.")
    parser.add_argument("--sector-map", help="Path to JSON or CSV sector map.")
    parser.add_argument("--start", default="2018-01-01", help="Backtest start date (YYYY-MM-DD).")
    parser.add_argument("--end", default="2026-04-15", help="Backtest end date (YYYY-MM-DD).")
    parser.add_argument("--interval", default="1d", help="Price bar interval.")
    parser.add_argument("--experiment-name", help="Experiment label. Defaults to the selected pipeline name.")
    parser.add_argument("--train-bars", type=int, default=252, help="Training bars per walk-forward fold.")
    parser.add_argument("--test-bars", type=int, default=63, help="Test bars per walk-forward fold.")
    parser.add_argument("--step-bars", type=int, default=63, help="Walk-forward step size.")
    parser.add_argument("--bars-per-year", type=int, default=252, help="Bars per year for annualization.")
    parser.add_argument(
        "--news-provider",
        nargs="+",
        choices=["local", "alphavantage", "benzinga"],
        help="One or more news sources to use before sentiment scoring.",
    )
    parser.add_argument("--news-file", nargs="*", help="One or more CSV/parquet files of raw news headlines.")
    parser.add_argument("--daily-sentiment-file", help="CSV or parquet of precomputed daily sentiment.")
    parser.add_argument("--news-api-key", help="API key for the selected remote news provider.")
    parser.add_argument("--alphavantage-api-key", help="API key for Alpha Vantage news.")
    parser.add_argument("--benzinga-api-key", help="API key for Benzinga news.")
    parser.add_argument("--news-topics", nargs="*", help="Optional topic filters for providers that support them.")
    parser.add_argument("--use-finbert", action="store_true", help="Use FinBERT for headline sentiment.")
    parser.add_argument("--local-finbert-only", action="store_true", help="Require FinBERT to already exist locally.")
    parser.add_argument("--artifact-root", default="artifacts/experiments", help="Experiment artifact directory.")
    parser.add_argument("--price-cache-dir", default="data/cache", help="Price parquet cache directory.")
    parser.add_argument("--sentiment-cache-dir", default="data/sentiment_cache", help="Sentiment cache directory.")
    parser.add_argument("--fast-window", type=int, default=20, help="Fast MA window for ma_cross.")
    parser.add_argument("--slow-window", type=int, default=80, help="Slow MA window for ma_cross.")
    parser.add_argument("--rsi-window", type=int, default=14, help="RSI window for rsi_mean_reversion.")
    parser.add_argument("--lower-entry", type=float, default=30.0, help="Long entry RSI threshold.")
    parser.add_argument("--upper-entry", type=float, default=70.0, help="Short entry RSI threshold.")
    parser.add_argument("--exit-level", type=float, default=50.0, help="RSI exit level.")
    parser.add_argument("--breakout-window", type=int, default=55, help="Lookback for Donchian breakouts.")
    parser.add_argument("--breakout-exit-window", type=int, default=20, help="Exit lookback for Donchian breakouts.")
    parser.add_argument("--strategy-cost-bps", type=float, default=2.0, help="Internal strategy turnover cost in bps.")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    experiment_name = args.experiment_name or args.pipeline

    if args.pipeline == "stat_arb":
        run_output = run_stat_arb_pipeline(
            sector_map_path=args.sector_map,
            start=args.start,
            end=args.end,
            interval=args.interval,
            experiment_name=experiment_name,
            price_cache_dir=args.price_cache_dir,
            sentiment_cache_dir=args.sentiment_cache_dir,
            artifact_root=args.artifact_root,
            news_provider_names=args.news_provider,
            news_files=args.news_file,
            daily_sentiment_file=args.daily_sentiment_file,
            use_finbert=args.use_finbert,
            local_finbert_only=args.local_finbert_only,
            news_api_key=args.news_api_key,
            alphavantage_api_key=args.alphavantage_api_key,
            benzinga_api_key=args.benzinga_api_key,
            news_topics=args.news_topics,
        )
    else:
        if not args.symbols:
            parser.error("--symbols is required for directional pipelines.")
        run_output = run_directional_pipeline(
            strategy_name=args.pipeline,
            symbols=args.symbols,
            start=args.start,
            end=args.end,
            interval=args.interval,
            experiment_name=experiment_name,
            price_cache_dir=args.price_cache_dir,
            artifact_root=args.artifact_root,
            train_bars=args.train_bars,
            test_bars=args.test_bars,
            step_bars=args.step_bars,
            bars_per_year=args.bars_per_year,
            fast_window=args.fast_window,
            slow_window=args.slow_window,
            rsi_window=args.rsi_window,
            lower_entry=args.lower_entry,
            upper_entry=args.upper_entry,
            exit_level=args.exit_level,
            breakout_window=args.breakout_window,
            breakout_exit_window=args.breakout_exit_window,
            strategy_cost_bps=args.strategy_cost_bps,
        )

    result: ExperimentResult = run_output["result"]
    visuals = run_output["visuals"]

    print(json.dumps(run_output["summary"], indent=2))
    print(f"Artifacts saved to: {result.artifact_dir}")
    for name, path in visuals.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
