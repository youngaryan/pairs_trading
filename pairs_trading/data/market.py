from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Sequence

import pandas as pd
import yfinance as yf


@dataclass(frozen=True)
class DataRequest:
    symbols: tuple[str, ...]
    start: str
    end: str
    interval: str = "1d"

    @classmethod
    def from_inputs(
        cls,
        symbols: Sequence[str],
        start: str | pd.Timestamp,
        end: str | pd.Timestamp,
        interval: str = "1d",
    ) -> "DataRequest":
        normalized_symbols = tuple(dict.fromkeys(symbols))
        return cls(
            symbols=normalized_symbols,
            start=str(pd.Timestamp(start).strftime("%Y-%m-%d")),
            end=str(pd.Timestamp(end).strftime("%Y-%m-%d")),
            interval=interval,
        )

    @property
    def cache_key(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True)
        return sha256(payload.encode("utf-8")).hexdigest()[:16]


class MarketDataProvider(ABC):
    @abstractmethod
    def get_close_prices(
        self,
        symbols: Sequence[str],
        start: str | pd.Timestamp,
        end: str | pd.Timestamp,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Return a close-price matrix indexed by timestamp with one column per symbol."""


class YahooFinanceProvider(MarketDataProvider):
    """Remote provider kept behind an interface so the rest of the code stays provider-agnostic."""

    def __init__(self, tz_cache_dir: str | Path = "data/yfinance_tz_cache") -> None:
        self.tz_cache_dir = Path(tz_cache_dir)
        self.tz_cache_dir.mkdir(parents=True, exist_ok=True)
        yf.set_tz_cache_location(str(self.tz_cache_dir))

    def get_close_prices(
        self,
        symbols: Sequence[str],
        start: str | pd.Timestamp,
        end: str | pd.Timestamp,
        interval: str = "1d",
    ) -> pd.DataFrame:
        request = DataRequest.from_inputs(symbols=symbols, start=start, end=end, interval=interval)
        raw = yf.download(
            list(request.symbols),
            start=request.start,
            end=request.end,
            interval=request.interval,
            progress=False,
            auto_adjust=False,
            group_by="column",
        )

        if raw.empty:
            raise ValueError(f"No price data returned for request: {request}")

        if isinstance(raw.columns, pd.MultiIndex):
            close_key = "Adj Close" if "Adj Close" in raw.columns.get_level_values(0) else "Close"
            close = raw[close_key].copy()
        else:
            column_name = "Adj Close" if "Adj Close" in raw.columns else "Close"
            close = raw[[column_name]].rename(columns={column_name: request.symbols[0]}).copy()

        close = close.loc[:, list(request.symbols)].dropna(how="all").sort_index()
        close.index = pd.DatetimeIndex(close.index).tz_localize(None)
        close.columns = [str(column) for column in close.columns]
        return close


class CachedParquetProvider(MarketDataProvider):
    """
    Cache provider that persists standardized close-price matrices to parquet.

    The cache key includes symbols, date range, and interval. That keeps the implementation
    straightforward and deterministic for research workflows.
    """

    def __init__(
        self,
        upstream: MarketDataProvider | None = None,
        cache_dir: str | Path = "data/cache",
    ) -> None:
        self.upstream = upstream
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _paths_for(self, request: DataRequest) -> tuple[Path, Path]:
        interval_dir = self.cache_dir / request.interval
        interval_dir.mkdir(parents=True, exist_ok=True)
        parquet_path = interval_dir / f"{request.cache_key}.parquet"
        meta_path = interval_dir / f"{request.cache_key}.json"
        return parquet_path, meta_path

    def _find_compatible_cache(self, request: DataRequest) -> pd.DataFrame | None:
        interval_dir = self.cache_dir / request.interval
        if not interval_dir.exists():
            return None

        request_start = pd.Timestamp(request.start)
        request_end = pd.Timestamp(request.end)
        requested_symbols = list(request.symbols)
        requested_symbol_set = set(requested_symbols)

        for meta_path in interval_dir.glob("*.json"):
            try:
                payload = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                continue

            cached_symbols = [str(symbol) for symbol in payload.get("symbols", [])]
            if not requested_symbol_set.issubset(set(cached_symbols)):
                continue

            cached_start = pd.Timestamp(payload.get("start"))
            cached_end = pd.Timestamp(payload.get("end"))
            if cached_start > request_start or cached_end < request_end:
                continue

            parquet_path = meta_path.with_suffix(".parquet")
            if not parquet_path.exists():
                continue

            cached = pd.read_parquet(parquet_path)
            missing_symbols = [symbol for symbol in requested_symbols if symbol not in cached.columns]
            if missing_symbols:
                continue

            cached = cached.loc[:, requested_symbols].sort_index()
            cached.index = pd.DatetimeIndex(cached.index).tz_localize(None)
            sliced = cached.loc[(cached.index >= request_start) & (cached.index < request_end)]
            if sliced.empty:
                continue
            return sliced

        return None

    def get_close_prices(
        self,
        symbols: Sequence[str],
        start: str | pd.Timestamp,
        end: str | pd.Timestamp,
        interval: str = "1d",
    ) -> pd.DataFrame:
        request = DataRequest.from_inputs(symbols=symbols, start=start, end=end, interval=interval)
        parquet_path, meta_path = self._paths_for(request)

        if parquet_path.exists():
            cached = pd.read_parquet(parquet_path)
            cached = cached.loc[:, list(request.symbols)].sort_index()
            cached.index = pd.DatetimeIndex(cached.index).tz_localize(None)
            return cached

        compatible = self._find_compatible_cache(request)
        if compatible is not None:
            return compatible

        if self.upstream is None:
            raise FileNotFoundError(f"Missing cache entry {parquet_path} and no upstream provider is configured.")

        fetched = self.upstream.get_close_prices(
            symbols=request.symbols,
            start=request.start,
            end=request.end,
            interval=request.interval,
        )

        fetched.to_parquet(parquet_path)
        with meta_path.open("w", encoding="utf-8") as handle:
            json.dump(asdict(request), handle, indent=2)

        return fetched
