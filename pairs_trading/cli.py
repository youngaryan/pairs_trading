from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

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
from .portfolio import PortfolioManager
from .research import PairScreenConfig
from .sentiment import FinBERTSentimentModel, SentimentConfig, build_best_available_sentiment_model
from .stat_arb import SectorStatArbPipeline, StatArbConfig
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
        providers.append(
            AlphaVantageNewsProvider(
                api_key=api_key,
                topics=news_topics,
            )
        )

    if "benzinga" in provider_names:
        api_key = benzinga_api_key or news_api_key or os.getenv("BENZINGA_API_KEY")
        if not api_key:
            raise ValueError("Benzinga news requires --benzinga-api-key, --news-api-key, or BENZINGA_API_KEY.")
        providers.append(BenzingaNewsProvider(api_key=api_key))

    if not providers:
        return None

    if use_finbert:
        model = FinBERTSentimentModel(local_files_only=local_finbert_only)
    else:
        model = build_best_available_sentiment_model()

    headline_provider = providers[0] if len(providers) == 1 else CompositeHeadlineProvider(providers)
    provider = CachedNewsSentimentProvider(
        headline_provider=headline_provider,
        sentiment_model=model,
        cache_dir=sentiment_cache_dir,
    )
    return provider.get_daily_sentiment(tickers=tickers, start=start, end=end)


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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the stat-arb walk-forward pipeline.")
    parser.add_argument("--sector-map", help="Path to JSON or CSV sector map.")
    parser.add_argument("--start", default="2018-01-01", help="Backtest start date (YYYY-MM-DD).")
    parser.add_argument("--end", default="2026-04-15", help="Backtest end date (YYYY-MM-DD).")
    parser.add_argument("--interval", default="1d", help="Price bar interval.")
    parser.add_argument("--experiment-name", default="serious_stat_arb", help="Experiment label.")
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
    parser.add_argument(
        "--news-topics",
        nargs="*",
        help="Optional topic filters for providers that support them, e.g. earnings technology.",
    )
    parser.add_argument("--use-finbert", action="store_true", help="Use FinBERT for headline sentiment.")
    parser.add_argument(
        "--local-finbert-only",
        action="store_true",
        help="Require FinBERT to already exist in the local model cache.",
    )
    parser.add_argument("--artifact-root", default="artifacts/experiments", help="Experiment artifact directory.")
    parser.add_argument("--price-cache-dir", default="data/cache", help="Price parquet cache directory.")
    parser.add_argument("--sentiment-cache-dir", default="data/sentiment_cache", help="Sentiment cache directory.")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    run_output = run_stat_arb_pipeline(
        sector_map_path=args.sector_map,
        start=args.start,
        end=args.end,
        interval=args.interval,
        experiment_name=args.experiment_name,
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

    result: ExperimentResult = run_output["result"]
    visuals = run_output["visuals"]

    print(json.dumps(run_output["summary"], indent=2))
    print(f"Artifacts saved to: {result.artifact_dir}")
    for name, path in visuals.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
