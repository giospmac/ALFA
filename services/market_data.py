from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
import streamlit as st
import yfinance as yf


class MarketDataError(Exception):
    """Raised when Yahoo Finance data cannot be retrieved or validated."""


UNAVAILABLE_REASON = "Indisponível via Yahoo Finance para este ativo."


@dataclass(frozen=True)
class MetricValue:
    label: str
    value: Any
    is_percent: bool = False
    is_currency: bool = False
    is_ratio: bool = False
    decimals: int = 2
    currency_code: str | None = None
    unavailable_reason: str = UNAVAILABLE_REASON

    @property
    def is_available(self) -> bool:
        return self.value is not None and not pd.isna(self.value)

    def formatted(self) -> str:
        if not self.is_available:
            return "N/A"

        numeric_value = float(self.value)
        if self.is_currency:
            currency = self.currency_code or "N/A"
            return f"{currency} {numeric_value:,.{self.decimals}f}"
        if self.is_percent:
            return f"{numeric_value * 100:.{self.decimals}f}%"
        if self.is_ratio:
            return f"{numeric_value:.{self.decimals}f}x"
        return f"{numeric_value:.{self.decimals}f}"


@dataclass(frozen=True)
class AssetSnapshot:
    ticker: str
    long_name: str
    currency: str
    metrics: dict[str, MetricValue]
    details: pd.DataFrame


def _coalesce_number(*values: Any) -> float | None:
    for value in values:
        if value is None or pd.isna(value):
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _extract_history_prices(history: pd.DataFrame) -> tuple[float | None, float | None]:
    if history.empty or "Close" not in history.columns:
        return None, None

    closes = pd.to_numeric(history["Close"], errors="coerce").dropna()
    if closes.empty:
        return None, None

    current_price = float(closes.iloc[-1])
    if len(closes) < 2:
        return current_price, None

    previous_close = float(closes.iloc[-2])
    if previous_close == 0:
        return current_price, None

    day_change_pct = (current_price / previous_close) - 1
    return current_price, day_change_pct


def _build_details_table(info: dict[str, Any]) -> pd.DataFrame:
    fields = {
        "Nome": info.get("longName") or info.get("shortName"),
        "Setor": info.get("sector"),
        "Indústria": info.get("industry"),
        "Moeda": info.get("currency"),
        "País": info.get("country"),
        "Bolsa": info.get("exchange"),
        "Tipo": info.get("quoteType"),
        "Market Cap": info.get("marketCap"),
        "Trailing PE": info.get("trailingPE"),
        "EV/EBITDA": info.get("enterpriseToEbitda"),
        "Margem EBITDA": info.get("ebitdaMargins"),
        "Crescimento Receita": info.get("revenueGrowth"),
        "Beta": info.get("beta"),
    }

    details = pd.DataFrame(
        [{"Campo": key, "Valor": "N/A" if value is None or pd.isna(value) else value} for key, value in fields.items()]
    )
    return details


@st.cache_data(ttl=60, show_spinner=False)
def fetch_asset_snapshot(ticker: str) -> AssetSnapshot:
    normalized_ticker = ticker.strip().upper()
    if not normalized_ticker:
        raise MarketDataError("Informe um ticker antes de analisar.")

    try:
        asset = yf.Ticker(normalized_ticker)
        info = asset.info or {}
        history = asset.history(period="5d", auto_adjust=False)
    except Exception as exc:
        raise MarketDataError("Falha ao consultar o Yahoo Finance no momento.") from exc

    has_metadata = bool(info) and info.get("quoteType") != "NONE"
    has_prices = not history.empty
    if not has_metadata and not has_prices:
        raise MarketDataError(f'Ticker "{normalized_ticker}" inválido ou sem dados disponíveis no Yahoo Finance.')

    long_name = info.get("longName") or info.get("shortName") or normalized_ticker
    currency = info.get("currency") or "BRL"

    current_price, day_change_pct = _extract_history_prices(history)
    current_price = _coalesce_number(
        info.get("currentPrice"),
        info.get("regularMarketPrice"),
        current_price,
    )
    previous_close = _coalesce_number(
        info.get("previousClose"),
        info.get("regularMarketPreviousClose"),
    )
    day_change = None
    if current_price is not None and previous_close not in (None, 0):
        day_change = current_price - previous_close

    metrics = {
        "current_price": MetricValue("Preço atual", current_price, is_currency=True, currency_code=currency),
        "beta": MetricValue("Beta", _coalesce_number(info.get("beta"))),
        "day_change": MetricValue("Variação do dia", day_change, is_currency=True, currency_code=currency),
        "day_change_pct": MetricValue("Variação do dia (%)", day_change_pct, is_percent=True),
        "pe_ratio": MetricValue("P/L", _coalesce_number(info.get("trailingPE"))),
        "ev_to_ebitda": MetricValue("EV/EBITDA", _coalesce_number(info.get("enterpriseToEbitda")), is_ratio=True),
        "market_cap": MetricValue(
            "Market Cap",
            _coalesce_number(info.get("marketCap")),
            is_currency=True,
            decimals=0,
            currency_code=currency,
        ),
        "ebitda_margin": MetricValue("Margem EBITDA", _coalesce_number(info.get("ebitdaMargins")), is_percent=True),
        "revenue_growth_yoy": MetricValue(
            "Crescimento de Receita (YoY)",
            _coalesce_number(info.get("revenueGrowth")),
            is_percent=True,
        ),
    }

    return AssetSnapshot(
        ticker=normalized_ticker,
        long_name=long_name,
        currency=currency,
        metrics=metrics,
        details=_build_details_table(info),
    )
