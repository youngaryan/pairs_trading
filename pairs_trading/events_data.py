from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
from urllib.request import Request, urlopen
from typing import Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EventRequest:
    tickers: tuple[str, ...]
    start: str
    end: str

    @classmethod
    def from_inputs(
        cls,
        tickers: Sequence[str],
        start: str | pd.Timestamp,
        end: str | pd.Timestamp,
    ) -> "EventRequest":
        normalized = tuple(dict.fromkeys(str(ticker).upper() for ticker in tickers))
        return cls(
            tickers=normalized,
            start=str(pd.Timestamp(start).strftime("%Y-%m-%d")),
            end=str(pd.Timestamp(end).strftime("%Y-%m-%d")),
        )

    @property
    def cache_key(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True)
        return sha256(payload.encode("utf-8")).hexdigest()[:16]


class EventProvider(ABC):
    @abstractmethod
    def get_events(
        self,
        tickers: Sequence[str],
        start: str | pd.Timestamp,
        end: str | pd.Timestamp,
    ) -> pd.DataFrame:
        """
        Return rows with:
        - timestamp
        - ticker
        - event_score
        Optional:
        - confidence
        - event_type
        - source
        - form
        """


class LocalEventFileProvider(EventProvider):
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def get_events(
        self,
        tickers: Sequence[str],
        start: str | pd.Timestamp,
        end: str | pd.Timestamp,
    ) -> pd.DataFrame:
        if self.path.suffix.lower() == ".parquet":
            frame = pd.read_parquet(self.path)
        else:
            frame = pd.read_csv(self.path)

        if frame.empty:
            return frame

        required = {"timestamp", "ticker", "event_score"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"Missing required event columns in {self.path}: {sorted(missing)}")

        filtered = frame.copy()
        filtered["timestamp"] = pd.to_datetime(filtered["timestamp"]).dt.tz_localize(None)
        filtered["ticker"] = filtered["ticker"].astype(str).str.upper()
        filtered["event_score"] = pd.to_numeric(filtered["event_score"], errors="coerce").fillna(0.0)
        if "confidence" not in filtered.columns:
            filtered["confidence"] = 1.0
        filtered["confidence"] = pd.to_numeric(filtered["confidence"], errors="coerce").fillna(1.0).clip(0.0, 1.0)
        if "event_type" not in filtered.columns:
            filtered["event_type"] = "file_event"
        if "source" not in filtered.columns:
            filtered["source"] = "local_file"
        if "form" not in filtered.columns:
            filtered["form"] = ""

        request = EventRequest.from_inputs(tickers=tickers, start=start, end=end)
        start_ts = pd.Timestamp(request.start)
        end_ts = pd.Timestamp(request.end)
        filtered = filtered[filtered["ticker"].isin(request.tickers)]
        filtered = filtered[(filtered["timestamp"] >= start_ts) & (filtered["timestamp"] <= end_ts)]
        return filtered.sort_values(["timestamp", "ticker"]).reset_index(drop=True)


class CachedEventProvider(EventProvider):
    def __init__(self, upstream: EventProvider | None = None, cache_dir: str | Path = "data/event_cache") -> None:
        self.upstream = upstream
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _paths_for(self, request: EventRequest) -> tuple[Path, Path]:
        parquet_path = self.cache_dir / f"{request.cache_key}.parquet"
        meta_path = self.cache_dir / f"{request.cache_key}.json"
        return parquet_path, meta_path

    def get_events(
        self,
        tickers: Sequence[str],
        start: str | pd.Timestamp,
        end: str | pd.Timestamp,
    ) -> pd.DataFrame:
        request = EventRequest.from_inputs(tickers=tickers, start=start, end=end)
        parquet_path, meta_path = self._paths_for(request)

        if parquet_path.exists():
            cached = pd.read_parquet(parquet_path)
            if not cached.empty and "timestamp" in cached.columns:
                cached["timestamp"] = pd.to_datetime(cached["timestamp"]).dt.tz_localize(None)
            return cached.sort_values(["timestamp", "ticker"]).reset_index(drop=True)

        if self.upstream is None:
            raise FileNotFoundError(f"Missing event cache entry {parquet_path} and no upstream provider is configured.")

        events = self.upstream.get_events(
            tickers=request.tickers,
            start=request.start,
            end=request.end,
        )
        events.to_parquet(parquet_path)
        meta_path.write_text(json.dumps(asdict(request), indent=2), encoding="utf-8")
        return events


class SecCompanyFactsEventProvider(EventProvider):
    """
    Builds EDGAR event scores from official SEC company facts.

    The provider derives a simple post-filing score from year-over-year changes in
    revenue and net income. It is intentionally conservative: the output is meant
    to be a clean event panel that an event-drift strategy can trade after the
    filing date, not a substitute for analyst-surprise data.
    """

    TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
    COMPANY_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"

    REVENUE_CONCEPTS = (
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "SalesRevenueNet",
        "Revenues",
    )
    EARNINGS_CONCEPTS = (
        "NetIncomeLoss",
        "ProfitLoss",
    )

    def __init__(
        self,
        user_agent: str,
        cache_dir: str | Path = "data/sec_cache",
        timeout_seconds: float = 30.0,
    ) -> None:
        if not user_agent or "@" not in user_agent:
            raise ValueError("SEC requests require a descriptive User-Agent with contact information.")
        self.user_agent = user_agent
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout_seconds = timeout_seconds

    def _fetch_json(self, url: str) -> dict:
        request = Request(url, headers={"User-Agent": self.user_agent})
        with urlopen(request, timeout=self.timeout_seconds) as response:
            payload = response.read().decode("utf-8")
        return json.loads(payload)

    def _ticker_map_path(self) -> Path:
        return self.cache_dir / "company_tickers.json"

    def _load_ticker_map(self) -> dict[str, str]:
        path = self._ticker_map_path()
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
        else:
            data = self._fetch_json(self.TICKER_MAP_URL)
            path.write_text(json.dumps(data), encoding="utf-8")

        mapping: dict[str, str] = {}
        for record in data.values():
            ticker = str(record.get("ticker", "")).upper()
            cik = str(record.get("cik_str", "")).strip()
            if ticker and cik:
                mapping[ticker] = cik.zfill(10)
        return mapping

    def _companyfacts_path(self, cik: str) -> Path:
        return self.cache_dir / "companyfacts" / f"CIK{cik}.json"

    def _load_companyfacts(self, cik: str) -> dict:
        path = self._companyfacts_path(cik)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))

        payload = self._fetch_json(self.COMPANY_FACTS_URL.format(cik=cik))
        path.write_text(json.dumps(payload), encoding="utf-8")
        return payload

    def _extract_series(self, payload: dict, concepts: Sequence[str]) -> tuple[pd.DataFrame, str | None]:
        facts = payload.get("facts", {}).get("us-gaap", {})
        for concept in concepts:
            fact = facts.get(concept)
            if not fact:
                continue

            rows: list[dict[str, object]] = []
            for unit_key, entries in fact.get("units", {}).items():
                if not str(unit_key).upper().startswith("USD"):
                    continue
                for entry in entries:
                    form = str(entry.get("form", ""))
                    filed = entry.get("filed")
                    fy = entry.get("fy")
                    fp = entry.get("fp")
                    value = entry.get("val")
                    if filed is None or fy is None or fp is None or value is None:
                        continue
                    if form not in {"10-Q", "10-Q/A", "10-K", "10-K/A"}:
                        continue
                    rows.append(
                        {
                            "filed": pd.Timestamp(filed),
                            "fy": int(fy),
                            "fp": str(fp),
                            "form": form,
                            "val": float(value),
                        }
                    )

            if not rows:
                continue

            frame = pd.DataFrame(rows)
            frame = frame.sort_values(["fy", "fp", "filed"]).drop_duplicates(["fy", "fp"], keep="last")
            return frame.reset_index(drop=True), concept

        return pd.DataFrame(columns=["filed", "fy", "fp", "form", "val"]), None

    @staticmethod
    def _yoy_growth(series: pd.DataFrame, value_column: str) -> pd.Series:
        previous = series.set_index(["fy", "fp"])[value_column].to_dict()
        growth = []
        for row in series.to_dict("records"):
            prior_value = previous.get((int(row["fy"]) - 1, str(row["fp"])))
            if prior_value in (None, 0):
                growth.append(float("nan"))
                continue
            growth.append(float(row[value_column]) / float(prior_value) - 1.0)
        return pd.Series(growth, index=series.index)

    def _build_company_events(self, ticker: str, payload: dict) -> pd.DataFrame:
        revenue, revenue_concept = self._extract_series(payload, self.REVENUE_CONCEPTS)
        earnings, earnings_concept = self._extract_series(payload, self.EARNINGS_CONCEPTS)

        if revenue.empty and earnings.empty:
            return pd.DataFrame(
                columns=[
                    "timestamp",
                    "ticker",
                    "event_score",
                    "confidence",
                    "event_type",
                    "source",
                    "form",
                    "revenue_yoy",
                    "earnings_yoy",
                ]
            )

        if not revenue.empty:
            revenue = revenue.copy()
            revenue["revenue_yoy"] = self._yoy_growth(revenue, "val")
        if not earnings.empty:
            earnings = earnings.copy()
            earnings["earnings_yoy"] = self._yoy_growth(earnings, "val")

        merged = pd.merge(
            revenue.rename(columns={"val": "revenue"}),
            earnings.rename(columns={"val": "earnings"}),
            on=["filed", "fy", "fp", "form"],
            how="outer",
        ).sort_values("filed")

        if "revenue_yoy" not in merged.columns:
            merged["revenue_yoy"] = float("nan")
        if "earnings_yoy" not in merged.columns:
            merged["earnings_yoy"] = float("nan")

        score_components = []
        if "revenue_yoy" in merged.columns:
            score_components.append(merged["revenue_yoy"].clip(-1.5, 1.5))
        if "earnings_yoy" in merged.columns:
            score_components.append(merged["earnings_yoy"].clip(-1.5, 1.5))

        if score_components:
            stacked = pd.concat(score_components, axis=1)
            merged["event_score"] = stacked.mean(axis=1, skipna=True).map(
                lambda value: 0.0 if pd.isna(value) else float(np.tanh(2.0 * value))
            )
            merged["confidence"] = stacked.notna().mean(axis=1).fillna(0.0)
        else:
            merged["event_score"] = 0.0
            merged["confidence"] = 0.0

        merged["timestamp"] = pd.to_datetime(merged["filed"]).dt.tz_localize(None)
        merged["ticker"] = ticker
        merged["event_type"] = "edgar_companyfacts"
        merged["source"] = "sec_companyfacts"
        merged["revenue_concept"] = revenue_concept or ""
        merged["earnings_concept"] = earnings_concept or ""

        result = merged[
            [
                "timestamp",
                "ticker",
                "event_score",
                "confidence",
                "event_type",
                "source",
                "form",
                "fy",
                "fp",
                "revenue_yoy",
                "earnings_yoy",
                "revenue_concept",
                "earnings_concept",
            ]
        ].copy()
        result = result[result["confidence"] > 0.0]
        return result.sort_values("timestamp").reset_index(drop=True)

    def get_events(
        self,
        tickers: Sequence[str],
        start: str | pd.Timestamp,
        end: str | pd.Timestamp,
    ) -> pd.DataFrame:
        request = EventRequest.from_inputs(tickers=tickers, start=start, end=end)
        ticker_map = self._load_ticker_map()

        events: list[pd.DataFrame] = []
        for ticker in request.tickers:
            cik = ticker_map.get(ticker)
            if cik is None:
                continue
            payload = self._load_companyfacts(cik)
            company_events = self._build_company_events(ticker, payload)
            if not company_events.empty:
                events.append(company_events)

        if not events:
            return pd.DataFrame(
                columns=[
                    "timestamp",
                    "ticker",
                    "event_score",
                    "confidence",
                    "event_type",
                    "source",
                    "form",
                ]
            )

        combined = pd.concat(events, axis=0, ignore_index=True)
        start_ts = pd.Timestamp(request.start)
        end_ts = pd.Timestamp(request.end)
        combined = combined[(combined["timestamp"] >= start_ts) & (combined["timestamp"] <= end_ts)]
        return combined.sort_values(["timestamp", "ticker"]).reset_index(drop=True)
