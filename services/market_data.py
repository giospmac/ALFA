from __future__ import annotations

from dataclasses import dataclass
import re
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


@dataclass(frozen=True)
class TickerComparisonResult:
    normalized_history: pd.DataFrame
    total_returns: dict[str, float]
    invalid_tickers: list[str]


def _coalesce_number(*values: Any) -> float | None:
    for value in values:
        if value is None or pd.isna(value):
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def normalize_market_ticker(ticker_input: str) -> str:
    ticker = str(ticker_input or "").strip().upper()
    if re.fullmatch(r"[A-Z]{4}\d{1,2}", ticker) and not ticker.endswith(".SA"):
        return f"{ticker}.SA"
    return ticker


def _extract_download_close_history(raw_history: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    if raw_history.empty:
        return pd.DataFrame()

    if isinstance(raw_history.columns, pd.MultiIndex):
        level_zero = raw_history.columns.get_level_values(0)
        level_last = raw_history.columns.get_level_values(-1)
        if "Close" in level_zero:
            close_history = raw_history.xs("Close", axis=1, level=0)
        elif "Close" in level_last:
            close_history = raw_history.xs("Close", axis=1, level=-1)
        else:
            return pd.DataFrame()
    else:
        if "Close" not in raw_history.columns:
            return pd.DataFrame()
        close_history = raw_history[["Close"]].copy()
        close_history.columns = tickers[:1]

    close_history = close_history.loc[:, ~close_history.columns.duplicated()]
    close_history.index = pd.to_datetime(close_history.index, errors="coerce")
    close_history = close_history[close_history.index.notna()].sort_index()
    return close_history.apply(pd.to_numeric, errors="coerce")


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


@st.cache_data(ttl=900, show_spinner=False)
def fetch_ticker_comparison(tickers: tuple[str, ...], period: str) -> TickerComparisonResult:
    normalized_tickers: list[str] = []
    seen: set[str] = set()
    for ticker in tickers:
        normalized_ticker = normalize_market_ticker(ticker)
        if not normalized_ticker or normalized_ticker in seen:
            continue
        normalized_tickers.append(normalized_ticker)
        seen.add(normalized_ticker)

    if not normalized_tickers:
        raise MarketDataError("Selecione pelo menos um ticker para comparar.")

    download_target: str | list[str]
    download_target = normalized_tickers[0] if len(normalized_tickers) == 1 else normalized_tickers

    try:
        raw_history = yf.download(
            download_target,
            period=period,
            auto_adjust=True,
            progress=False,
            threads=False,
        )
    except Exception as exc:
        raise MarketDataError("Falha ao consultar as series historicas no Yahoo Finance.") from exc

    close_history = _extract_download_close_history(raw_history, normalized_tickers)
    if close_history.empty:
        raise MarketDataError("Nao foi possivel montar o historico dos tickers selecionados.")

    close_history = close_history.ffill().dropna(how="all")
    valid_tickers: list[str] = []
    invalid_tickers: list[str] = []

    for ticker in normalized_tickers:
        if ticker not in close_history.columns:
            invalid_tickers.append(ticker)
            continue
        series = pd.to_numeric(close_history[ticker], errors="coerce").dropna()
        if len(series) < 2:
            invalid_tickers.append(ticker)
            continue
        valid_tickers.append(ticker)

    if not valid_tickers:
        raise MarketDataError("Nenhum ticker retornou historico suficiente para comparacao.")

    comparison_history = close_history[valid_tickers].dropna()
    if len(comparison_history) < 2:
        comparison_history = close_history[valid_tickers].ffill().bfill().dropna(how="all")
    if len(comparison_history) < 2:
        raise MarketDataError("Os tickers selecionados nao possuem uma janela comum suficiente para comparacao.")

    base_prices = comparison_history.iloc[0].replace(0, pd.NA)
    normalized_history = comparison_history.divide(base_prices).dropna(how="all")
    total_returns = {
        ticker: float(normalized_history[ticker].dropna().iloc[-1] - 1.0)
        for ticker in valid_tickers
        if not normalized_history[ticker].dropna().empty
    }

    return TickerComparisonResult(
        normalized_history=normalized_history,
        total_returns=total_returns,
        invalid_tickers=invalid_tickers,
    )


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
