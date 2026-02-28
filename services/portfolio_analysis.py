from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import streamlit as st
import yfinance as yf


class PortfolioAnalysisError(Exception):
    """Raised when an ad hoc portfolio cannot be analyzed."""


DEFAULT_PORTFOLIO_ROWS = [
    {"Ticker": "AAPL", "Peso (%)": 50.0},
    {"Ticker": "MSFT", "Peso (%)": 50.0},
]


@dataclass(frozen=True)
class PortfolioAnalysisResult:
    allocations: pd.DataFrame
    price_history: pd.DataFrame
    normalized_history: pd.DataFrame
    portfolio_history: pd.DataFrame
    invalid_tickers: list[str]
    total_input_weight: float


def _extract_close_prices(raw_history: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    if raw_history.empty:
        return pd.DataFrame()

    if isinstance(raw_history.columns, pd.MultiIndex):
        if "Close" not in raw_history.columns.get_level_values(-1):
            return pd.DataFrame()
        close_history = raw_history.xs("Close", axis=1, level=-1)
    else:
        if "Close" not in raw_history.columns:
            return pd.DataFrame()
        close_history = raw_history[["Close"]].copy()
        close_history.columns = tickers[:1]

    close_history = close_history.loc[:, ~close_history.columns.duplicated()]
    return close_history.apply(pd.to_numeric, errors="coerce").dropna(how="all")


def _normalize_allocations(portfolio_df: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    working_df = portfolio_df.copy()
    working_df["Ticker"] = working_df["Ticker"].astype(str).str.strip().str.upper()
    working_df["Peso (%)"] = pd.to_numeric(working_df["Peso (%)"], errors="coerce")
    working_df = working_df.dropna(subset=["Ticker", "Peso (%)"])
    working_df = working_df[working_df["Ticker"] != ""]
    working_df = working_df[working_df["Peso (%)"] > 0]

    if working_df.empty:
        raise PortfolioAnalysisError("Adicione pelo menos um ticker com peso maior que zero.")

    aggregated_df = working_df.groupby("Ticker", as_index=False)["Peso (%)"].sum()
    total_input_weight = float(aggregated_df["Peso (%)"].sum())
    if total_input_weight <= 0:
        raise PortfolioAnalysisError("A soma dos pesos do portfólio precisa ser maior que zero.")

    aggregated_df["Peso Normalizado"] = aggregated_df["Peso (%)"] / total_input_weight
    return aggregated_df, total_input_weight


@st.cache_data(ttl=60, show_spinner=False)
def analyze_portfolio(portfolio_rows: tuple[tuple[str, float], ...], period: str) -> PortfolioAnalysisResult:
    portfolio_df = pd.DataFrame(portfolio_rows, columns=["Ticker", "Peso (%)"])
    allocations, total_input_weight = _normalize_allocations(portfolio_df)

    tickers = allocations["Ticker"].tolist()
    try:
        raw_history = yf.download(
            tickers=tickers,
            period=period,
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=False,
        )
    except Exception as exc:
        raise PortfolioAnalysisError("Falha ao buscar preços históricos no Yahoo Finance.") from exc

    close_history = _extract_close_prices(raw_history, tickers)
    if close_history.empty:
        raise PortfolioAnalysisError("O Yahoo Finance não retornou preços históricos para os tickers informados.")

    valid_tickers = [ticker for ticker in tickers if ticker in close_history.columns]
    invalid_tickers = [ticker for ticker in tickers if ticker not in valid_tickers]
    if not valid_tickers:
        raise PortfolioAnalysisError("Nenhum ticker válido retornou histórico utilizável no Yahoo Finance.")

    close_history = close_history[valid_tickers].dropna(how="all").ffill().dropna(how="all")
    if close_history.empty:
        raise PortfolioAnalysisError("Os históricos retornados vieram vazios após limpeza dos dados.")

    valid_allocations = allocations[allocations["Ticker"].isin(valid_tickers)].copy()
    valid_allocations["Peso Normalizado"] = valid_allocations["Peso Normalizado"] / valid_allocations["Peso Normalizado"].sum()

    latest_prices = close_history.ffill().iloc[-1]
    first_prices = close_history.bfill().iloc[0]
    total_returns = (latest_prices / first_prices) - 1

    valid_allocations["Preço Atual"] = valid_allocations["Ticker"].map(latest_prices.to_dict())
    valid_allocations["Retorno no Período"] = valid_allocations["Ticker"].map(total_returns.to_dict())

    normalized_history = close_history.divide(first_prices).multiply(100)
    weights = valid_allocations.set_index("Ticker")["Peso Normalizado"].reindex(valid_tickers)
    portfolio_index = normalized_history[valid_tickers].multiply(weights, axis=1).sum(axis=1)
    portfolio_history = pd.DataFrame({"Portfólio": portfolio_index})

    return PortfolioAnalysisResult(
        allocations=valid_allocations,
        price_history=close_history,
        normalized_history=normalized_history,
        portfolio_history=portfolio_history,
        invalid_tickers=invalid_tickers,
        total_input_weight=total_input_weight,
    )
