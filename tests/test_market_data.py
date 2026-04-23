from __future__ import annotations

from pathlib import Path
import unittest

import pandas as pd

from pairs_trading.market_data import CachedParquetProvider, MarketDataProvider
from tests.common import fresh_test_dir


class DummyProvider(MarketDataProvider):
    def __init__(self, data: pd.DataFrame) -> None:
        self.data = data
        self.calls = 0

    def get_close_prices(self, symbols, start, end, interval="1d") -> pd.DataFrame:
        self.calls += 1
        return self.data.loc[:, list(symbols)].copy()


class MarketDataCacheTests(unittest.TestCase):
    def test_cached_parquet_provider_uses_cache_after_first_fetch(self) -> None:
        index = pd.date_range("2024-01-01", periods=3, freq="D")
        data = pd.DataFrame({"AAA": [1.0, 1.1, 1.2], "BBB": [2.0, 2.1, 2.2]}, index=index)
        upstream = DummyProvider(data=data)
        cache_dir = fresh_test_dir("artifacts/test_cache/market_data")

        provider = CachedParquetProvider(upstream=upstream, cache_dir=cache_dir)
        first = provider.get_close_prices(["AAA", "BBB"], "2024-01-01", "2024-01-04")
        self.assertEqual(upstream.calls, 1)
        self.assertTrue(any(Path(cache_dir).rglob("*.parquet")))

        upstream.data = pd.DataFrame({"AAA": [9.0, 9.0, 9.0], "BBB": [9.0, 9.0, 9.0]}, index=index)
        second = provider.get_close_prices(["AAA", "BBB"], "2024-01-01", "2024-01-04")

        self.assertEqual(upstream.calls, 1)
        pd.testing.assert_frame_equal(first, second, check_freq=False)


if __name__ == "__main__":
    unittest.main()
