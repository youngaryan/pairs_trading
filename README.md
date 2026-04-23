# Quant Research App

This repository is structured as an early-stage professional quant research app rather than a single strategy script.

The architecture follows a modular flow:
- `pairs_trading/framework.py`: shared strategy interfaces and standardized outputs
- `pairs_trading/strategies/`: reusable alpha models
- `pairs_trading/pipelines/`: orchestration layers that turn strategy outputs into portfolios
- `pairs_trading/portfolio.py`: portfolio construction and leverage control
- `pairs_trading/backtesting.py`: walk-forward backtests, costs, metrics, and artifact persistence
- `pairs_trading/market_data.py` and `pairs_trading/news_data.py`: provider interfaces plus parquet caching
- `pairs_trading/sentiment.py`: finance-specific sentiment scoring and overlays
- `pairs_trading/visualization.py`: charts and HTML reports

## Run From Source

General help:

```powershell
.\.venv\Scripts\python.exe -m pairs_trading --help
```

Stat-arb pipeline:

```powershell
.\.venv\Scripts\python.exe -m pairs_trading `
  --pipeline stat_arb `
  --experiment-name sector_stat_arb
```

Stat-arb with local news sentiment:

```powershell
.\.venv\Scripts\python.exe -m pairs_trading `
  --pipeline stat_arb `
  --news-provider local `
  --news-file data\news\headlines.csv `
  --use-finbert `
  --experiment-name stat_arb_finbert
```

Moving-average crossover research:

```powershell
.\.venv\Scripts\python.exe -m pairs_trading `
  --pipeline ma_cross `
  --symbols AAPL MSFT NVDA `
  --fast-window 20 `
  --slow-window 80 `
  --experiment-name ma_cross_equities
```

RSI mean-reversion research:

```powershell
.\.venv\Scripts\python.exe -m pairs_trading `
  --pipeline rsi_mean_reversion `
  --symbols SPY QQQ IWM `
  --rsi-window 14 `
  --lower-entry 30 `
  --upper-entry 70 `
  --exit-level 50 `
  --experiment-name rsi_reversion
```

Donchian breakout research:

```powershell
.\.venv\Scripts\python.exe -m pairs_trading `
  --pipeline donchian_breakout `
  --symbols GLD TLT XLE `
  --breakout-window 55 `
  --breakout-exit-window 20 `
  --experiment-name donchian_macro
```

The root launcher still works too:

```powershell
.\.venv\Scripts\python.exe run_pipeline.py --pipeline ma_cross --symbols AAPL MSFT NVDA
```

## Output

Each run writes an experiment folder under `artifacts/experiments/<timestamp>_<name>/` with:
- `summary.json`
- `diagnostics.json`
- parquet outputs for folds and equity
- a `visuals/` folder with charts and an HTML report

## Tests

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## Packaging

This repo includes a `pyproject.toml` so it can grow into an installable internal research package. Once installed in editable mode, you can also run:

```powershell
pairs-trading --pipeline ma_cross --symbols AAPL MSFT NVDA
```
