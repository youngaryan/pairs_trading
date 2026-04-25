from __future__ import annotations

import unittest

import pandas as pd

from pairs_trading.data.news import AlphaVantageNewsProvider, BenzingaNewsProvider


class StubAlphaVantageProvider(AlphaVantageNewsProvider):
    def __init__(self) -> None:
        super().__init__(api_key="demo", topics=["earnings"], sort="LATEST", limit=50)
        self.captured_params: dict[str, str] | None = None

    def _fetch_json(self, url: str, params: dict[str, str], headers=None):
        self.captured_params = params
        return {
            "feed": [
                {
                    "title": "Apple beats estimates",
                    "summary": "Guidance was raised.",
                    "time_published": "20240102T143000",
                    "source": "Example",
                    "url": "https://example.com/aapl",
                    "overall_sentiment_score": 0.41,
                    "overall_sentiment_label": "Bullish",
                    "ticker_sentiment": [
                        {
                            "ticker": "AAPL",
                            "relevance_score": "0.91",
                        },
                        {
                            "ticker": "MSFT",
                            "relevance_score": "0.20",
                        },
                    ],
                }
            ]
        }


class StubBenzingaProvider(BenzingaNewsProvider):
    def __init__(self) -> None:
        super().__init__(api_key="demo", display_output="abstract", page_size=2, max_pages=3)
        self.pages_requested: list[int] = []

    def _fetch_json(self, url: str, params: dict[str, str], headers=None):
        self.pages_requested.append(int(params["page"]))
        if params["page"] == "0":
            return [
                {
                    "title": "Nvidia news",
                    "teaser": "Chip demand strong",
                    "body": "",
                    "created": "Mon, 01 Jan 2024 13:35:14 -0400",
                    "url": "https://example.com/nvda",
                    "author": "Benzinga Insights",
                    "channels": [{"name": "Technology"}],
                    "stocks": [{"name": "NVDA"}, {"name": "AMD"}],
                },
                {
                    "title": "Ignored ticker",
                    "teaser": "No match",
                    "body": "",
                    "created": "Mon, 01 Jan 2024 14:00:00 -0400",
                    "url": "https://example.com/other",
                    "author": "Benzinga Insights",
                    "channels": [{"name": "Markets"}],
                    "stocks": [{"name": "SPY"}],
                },
            ]
        return []


class RemoteNewsAdapterTests(unittest.TestCase):
    def test_alpha_vantage_adapter_builds_expected_request_and_rows(self) -> None:
        provider = StubAlphaVantageProvider()
        headlines = provider.get_headlines(["AAPL"], "2024-01-01", "2024-01-03")

        self.assertIsNotNone(provider.captured_params)
        assert provider.captured_params is not None
        self.assertEqual(provider.captured_params["function"], "NEWS_SENTIMENT")
        self.assertEqual(provider.captured_params["tickers"], "AAPL")
        self.assertEqual(provider.captured_params["topics"], "earnings")
        self.assertEqual(provider.captured_params["time_from"], "20240101T0000")
        self.assertEqual(provider.captured_params["time_to"], "20240103T2359")

        self.assertEqual(len(headlines), 1)
        self.assertEqual(headlines.loc[0, "ticker"], "AAPL")
        self.assertAlmostEqual(float(headlines.loc[0, "relevance"]), 0.91, places=6)
        self.assertIn("Apple beats estimates", headlines.loc[0, "headline"])

    def test_benzinga_adapter_paginates_and_filters_requested_tickers(self) -> None:
        provider = StubBenzingaProvider()
        headlines = provider.get_headlines(["NVDA"], "2024-01-01", "2024-01-02")

        self.assertEqual(provider.pages_requested, [0, 1])
        self.assertEqual(len(headlines), 1)
        self.assertEqual(headlines.loc[0, "ticker"], "NVDA")
        self.assertEqual(headlines.loc[0, "source"], "Benzinga")
        self.assertIn("Chip demand strong", headlines.loc[0, "headline"])
        self.assertEqual(headlines.loc[0, "channels"], "Technology")
        self.assertTrue(pd.notna(headlines.loc[0, "timestamp"]))


if __name__ == "__main__":
    unittest.main()
