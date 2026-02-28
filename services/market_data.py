from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

import numpy as np
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
    metric_history: pd.DataFrame
    metric_labels: dict[str, str]


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


def _safe_ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator in (None, 0):
        return None
    return numerator / denominator


def _safe_frame(frame: Any) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return pd.DataFrame()

    normalized = frame.copy()
    normalized.columns = pd.to_datetime(normalized.columns, errors="coerce")
    normalized = normalized.loc[:, normalized.columns.notna()]
    if normalized.empty:
        return pd.DataFrame()

    normalized = normalized.loc[:, ~normalized.columns.duplicated()].sort_index(axis=1)
    return normalized


def _statement_series(frame: pd.DataFrame, row_names: list[str]) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype=float)

    for row_name in row_names:
        if row_name not in frame.index:
            continue
        series = pd.to_numeric(frame.loc[row_name], errors="coerce")
        series.index = pd.to_datetime(series.index, errors="coerce")
        series = series[series.index.notna()].sort_index()
        if not series.empty:
            return series
    return pd.Series(dtype=float)


def _value_at_or_before(series: pd.Series, date: pd.Timestamp) -> float | None:
    if series.empty:
        return None
    eligible = series[series.index <= date].dropna()
    if eligible.empty:
        return None
    return _coalesce_number(eligible.iloc[-1])


def _ttm_value(series: pd.Series, date: pd.Timestamp, periods: int = 4) -> float | None:
    if series.empty:
        return None
    eligible = series[series.index <= date].dropna().sort_index()
    if eligible.empty:
        return None
    return _coalesce_number(eligible.tail(periods).sum())


def _price_on_or_before(close_prices: pd.Series, date: pd.Timestamp) -> float | None:
    if close_prices.empty:
        return None
    eligible = close_prices[close_prices.index <= date].dropna()
    if eligible.empty:
        return None
    return _coalesce_number(eligible.iloc[-1])


def _weekly_price_change_at(close_prices: pd.Series, date: pd.Timestamp) -> float | None:
    if close_prices.empty:
        return None
    eligible = close_prices[close_prices.index <= date].dropna()
    if len(eligible) < 6:
        return None
    current_price = _coalesce_number(eligible.iloc[-1])
    prior_price = _coalesce_number(eligible.iloc[-6])
    return _safe_ratio(current_price, prior_price) - 1 if current_price is not None and prior_price not in (None, 0) else None


def _rolling_beta_series(asset_prices: pd.Series, benchmark_prices: pd.Series) -> pd.Series:
    aligned = pd.concat(
        [
            pd.to_numeric(asset_prices, errors="coerce").rename("asset"),
            pd.to_numeric(benchmark_prices, errors="coerce").rename("benchmark"),
        ],
        axis=1,
    ).dropna()
    if len(aligned) < 60:
        return pd.Series(dtype=float)

    returns = aligned.pct_change().dropna()
    if len(returns) < 60:
        return pd.Series(dtype=float)

    rolling_cov = returns["asset"].rolling(window=126).cov(returns["benchmark"])
    rolling_var = returns["benchmark"].rolling(window=126).var()
    beta = rolling_cov.divide(rolling_var.replace(0, np.nan))
    return beta.dropna()


def _metric_label_map() -> dict[str, str]:
    return {
        "current_price": "Preço atual",
        "weekly_price_change_pct": "Variação semanal (%)",
        "market_cap": "Valor de mercado",
        "enterprise_value": "Enterprise Value",
        "roic": "ROIC",
        "ebitda_margin": "Margem EBITDA",
        "ev_to_ebitda": "EV/EBITDA",
        "dividend_yield": "Dividend Yield",
        "pe_ratio": "P/L",
        "financial_leverage": "Alavancagem financeira",
        "beta": "Beta",
    }


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


def _extract_single_close_series(raw_history: pd.DataFrame, ticker: str) -> pd.Series:
    close_history = _extract_download_close_history(raw_history, [ticker])
    if close_history.empty:
        return pd.Series(dtype=float)

    first_column = close_history.columns[0]
    series = pd.to_numeric(close_history[first_column], errors="coerce").dropna()
    series.index = pd.to_datetime(series.index, errors="coerce")
    return series[series.index.notna()].sort_index()


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
        "Enterprise Value": info.get("enterpriseValue"),
        "Trailing PE": info.get("trailingPE"),
        "Dividend Yield": info.get("dividendYield"),
        "Debt / Equity": info.get("debtToEquity"),
        "EV/EBITDA": info.get("enterpriseToEbitda"),
        "Margem EBITDA": info.get("ebitdaMargins"),
        "Crescimento Receita": info.get("revenueGrowth"),
        "Beta": info.get("beta"),
    }

    details = pd.DataFrame(
        [{"Campo": key, "Valor": "N/A" if value is None or pd.isna(value) else value} for key, value in fields.items()]
    )
    return details


def _build_metric_history(asset: yf.Ticker, ticker: str, info: dict[str, Any]) -> pd.DataFrame:
    price_history = asset.history(period="10y", auto_adjust=False)
    close_prices = _extract_single_close_series(price_history, ticker)

    income_stmt = _safe_frame(getattr(asset, "quarterly_income_stmt", pd.DataFrame()))
    if income_stmt.empty:
        income_stmt = _safe_frame(getattr(asset, "income_stmt", pd.DataFrame()))

    balance_sheet = _safe_frame(getattr(asset, "quarterly_balance_sheet", pd.DataFrame()))
    if balance_sheet.empty:
        balance_sheet = _safe_frame(getattr(asset, "balance_sheet", pd.DataFrame()))

    statement_dates = sorted(set(income_stmt.columns.tolist() + balance_sheet.columns.tolist()))
    if not statement_dates:
        return pd.DataFrame()

    revenue_series = _statement_series(income_stmt, ["Total Revenue", "Operating Revenue"])
    ebitda_series = _statement_series(income_stmt, ["EBITDA", "Normalized EBITDA"])
    operating_income_series = _statement_series(income_stmt, ["Operating Income"])
    pretax_income_series = _statement_series(income_stmt, ["Pretax Income", "Pre Tax Income"])
    tax_series = _statement_series(income_stmt, ["Tax Provision", "Tax Rate For Calcs"])
    net_income_series = _statement_series(income_stmt, ["Net Income", "Net Income Common Stockholders"])

    debt_series = _statement_series(balance_sheet, ["Total Debt", "Total Borrowings", "Long Term Debt And Capital Lease Obligation"])
    cash_series = _statement_series(
        balance_sheet,
        [
            "Cash And Cash Equivalents",
            "Cash Cash Equivalents And Short Term Investments",
            "Cash And Short Term Investments",
        ],
    )
    equity_series = _statement_series(
        balance_sheet,
        ["Stockholders Equity", "Total Equity Gross Minority Interest", "Common Stock Equity"],
    )
    minority_interest_series = _statement_series(balance_sheet, ["Minority Interest", "Minority Interests"])
    preferred_equity_series = _statement_series(balance_sheet, ["Preferred Stock Equity", "Preferred Securities Outside Stock Equity"])

    shares_outstanding = _coalesce_number(info.get("sharesOutstanding"), info.get("impliedSharesOutstanding"))

    dividends = getattr(asset, "dividends", pd.Series(dtype=float))
    dividends = pd.to_numeric(dividends, errors="coerce").dropna()
    dividends.index = pd.to_datetime(dividends.index, errors="coerce")
    dividends = dividends[dividends.index.notna()].sort_index()

    benchmark_symbol = "^BVSP" if ticker.endswith(".SA") else "^GSPC"
    try:
        benchmark_history = yf.download(
            benchmark_symbol,
            period="10y",
            auto_adjust=True,
            progress=False,
            threads=False,
        )
    except Exception:
        benchmark_history = pd.DataFrame()
    benchmark_close = _extract_single_close_series(benchmark_history, benchmark_symbol)
    beta_series = _rolling_beta_series(close_prices, benchmark_close)

    rows: list[dict[str, Any]] = []
    for date in statement_dates[-12:]:
        date = pd.Timestamp(date)
        price = _price_on_or_before(close_prices, date)
        weekly_change = _weekly_price_change_at(close_prices, date)

        revenue_ttm = _ttm_value(revenue_series, date)
        ebitda_ttm = _ttm_value(ebitda_series, date)
        operating_income_ttm = _ttm_value(operating_income_series, date)
        pretax_income_ttm = _ttm_value(pretax_income_series, date)
        tax_ttm = _ttm_value(tax_series, date)
        net_income_ttm = _ttm_value(net_income_series, date)

        total_debt = _value_at_or_before(debt_series, date)
        cash = _value_at_or_before(cash_series, date) or 0.0
        equity = _value_at_or_before(equity_series, date)
        minority_interest = _value_at_or_before(minority_interest_series, date) or 0.0
        preferred_equity = _value_at_or_before(preferred_equity_series, date) or 0.0

        market_cap = (price * shares_outstanding) if price is not None and shares_outstanding is not None else None
        enterprise_value = None
        if market_cap is not None:
            enterprise_value = market_cap + (total_debt or 0.0) + minority_interest + preferred_equity - cash

        effective_tax_rate = _safe_ratio(tax_ttm, pretax_income_ttm)
        if effective_tax_rate is None or not np.isfinite(effective_tax_rate):
            effective_tax_rate = 0.25
        effective_tax_rate = float(min(max(effective_tax_rate, 0.0), 0.5))

        invested_capital = None
        if equity is not None:
            invested_capital = (total_debt or 0.0) + equity - cash

        trailing_dividends = None
        if not dividends.empty:
            trailing_dividends = _coalesce_number(dividends[(dividends.index <= date) & (dividends.index > date - pd.Timedelta(days=365))].sum())

        beta_value = _value_at_or_before(beta_series, date)

        rows.append(
            {
                "date": date,
                "weekly_price_change_pct": weekly_change,
                "market_cap": market_cap,
                "enterprise_value": enterprise_value,
                "roic": _safe_ratio(operating_income_ttm * (1 - effective_tax_rate), invested_capital) if operating_income_ttm is not None else None,
                "ebitda_margin": _safe_ratio(ebitda_ttm, revenue_ttm),
                "ev_to_ebitda": _safe_ratio(enterprise_value, ebitda_ttm),
                "dividend_yield": _safe_ratio(trailing_dividends, price),
                "pe_ratio": _safe_ratio(market_cap, net_income_ttm),
                "financial_leverage": _safe_ratio(total_debt, equity),
                "beta": beta_value,
            }
        )

    history_df = pd.DataFrame(rows).set_index("date") if rows else pd.DataFrame()
    if history_df.empty:
        return pd.DataFrame()

    history_df.index = pd.to_datetime(history_df.index, errors="coerce")
    history_df = history_df[history_df.index.notna()].sort_index()
    return history_df.dropna(how="all")


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
    normalized_ticker = normalize_market_ticker(ticker)
    if not normalized_ticker:
        raise MarketDataError("Informe um ticker antes de analisar.")

    try:
        asset = yf.Ticker(normalized_ticker)
        info = asset.info or {}
        history = asset.history(period="1mo", auto_adjust=False)
    except Exception as exc:
        raise MarketDataError("Falha ao consultar o Yahoo Finance no momento.") from exc

    has_metadata = bool(info) and info.get("quoteType") != "NONE"
    has_prices = not history.empty
    if not has_metadata and not has_prices:
        raise MarketDataError(f'Ticker "{normalized_ticker}" inválido ou sem dados disponíveis no Yahoo Finance.')

    long_name = info.get("longName") or info.get("shortName") or normalized_ticker
    currency = info.get("currency") or "BRL"

    current_price, _ = _extract_history_prices(history)
    weekly_price_change_pct = _weekly_price_change_at(_extract_single_close_series(history, normalized_ticker), pd.Timestamp.now())
    current_price = _coalesce_number(
        info.get("currentPrice"),
        info.get("regularMarketPrice"),
        current_price,
    )
    metric_history = _build_metric_history(asset, normalized_ticker, info)
    metric_labels = _metric_label_map()
    latest_history = metric_history.iloc[-1] if not metric_history.empty else pd.Series(dtype=float)

    enterprise_value = _coalesce_number(
        info.get("enterpriseValue"),
        latest_history.get("enterprise_value") if not latest_history.empty else None,
    )
    financial_leverage = _coalesce_number(
        latest_history.get("financial_leverage") if not latest_history.empty else None,
        _coalesce_number(info.get("debtToEquity")) / 100 if _coalesce_number(info.get("debtToEquity")) is not None else None,
    )
    roic = _coalesce_number(latest_history.get("roic") if not latest_history.empty else None)
    dividend_yield = _coalesce_number(
        info.get("dividendYield"),
        latest_history.get("dividend_yield") if not latest_history.empty else None,
    )
    beta = _coalesce_number(
        info.get("beta"),
        latest_history.get("beta") if not latest_history.empty else None,
    )
    pe_ratio = _coalesce_number(
        info.get("trailingPE"),
        latest_history.get("pe_ratio") if not latest_history.empty else None,
    )
    ev_to_ebitda = _coalesce_number(
        info.get("enterpriseToEbitda"),
        latest_history.get("ev_to_ebitda") if not latest_history.empty else None,
    )
    market_cap = _coalesce_number(
        info.get("marketCap"),
        latest_history.get("market_cap") if not latest_history.empty else None,
    )
    ebitda_margin = _coalesce_number(
        info.get("ebitdaMargins"),
        latest_history.get("ebitda_margin") if not latest_history.empty else None,
    )

    metrics = {
        "current_price": MetricValue("Preço atual", current_price, is_currency=True, currency_code=currency),
        "beta": MetricValue("Beta", beta),
        "weekly_price_change_pct": MetricValue("Variação semanal (%)", weekly_price_change_pct, is_percent=True),
        "pe_ratio": MetricValue("P/L", pe_ratio),
        "ev_to_ebitda": MetricValue("EV/EBITDA", ev_to_ebitda, is_ratio=True),
        "market_cap": MetricValue(
            "Valor de mercado",
            market_cap,
            is_currency=True,
            decimals=0,
            currency_code=currency,
        ),
        "enterprise_value": MetricValue(
            "Enterprise Value",
            enterprise_value,
            is_currency=True,
            decimals=0,
            currency_code=currency,
        ),
        "ebitda_margin": MetricValue("Margem EBITDA", ebitda_margin, is_percent=True),
        "roic": MetricValue("ROIC", roic, is_percent=True),
        "dividend_yield": MetricValue(
            "Dividend Yield",
            dividend_yield,
            is_percent=True,
        ),
        "financial_leverage": MetricValue("Alavancagem financeira", financial_leverage, is_ratio=True),
    }

    return AssetSnapshot(
        ticker=normalized_ticker,
        long_name=long_name,
        currency=currency,
        metrics=metrics,
        details=_build_details_table(info),
        metric_history=metric_history,
        metric_labels=metric_labels,
    )
