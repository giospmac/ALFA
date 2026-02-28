from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
import re
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
import streamlit as st
import yfinance as yf
from scipy.stats import norm

from core.portfolio_repository import DEFAULT_PORTFOLIO_COLUMNS

try:
    from bcb import sgs
except ImportError:
    sgs = None


TRADING_DAYS_PER_YEAR = 252
ANNUAL_CDI_RATE = 0.1075
TREASURY_URL = (
    "https://www.tesourotransparente.gov.br/ckan/dataset/"
    "df56aa42-484a-4a59-8184-7676580c81e3/resource/"
    "796d2059-14e9-44e3-80c9-2d9e30b405c1/download/PrecoTaxaTesouroDireto.csv"
)
TREASURY_TYPE_MAP = {
    "Tesouro Educa+": "Educa",
    "Tesouro IGPM+ com Juros Semestrais": "IGPM",
    "Tesouro IPCA+": "IPCA",
    "Tesouro IPCA+ com Juros Semestrais": "IPCA",
    "Tesouro Prefixado": "Prefixado",
    "Tesouro Prefixado com Juros Semestrais": "Prefixado",
    "Tesouro Renda+ Aposentadoria Extra": "Renda",
    "Tesouro Selic": "Selic",
}
TREASURY_TYPES = list(TREASURY_TYPE_MAP.keys())


class PortfolioAnalyticsError(Exception):
    """Raised when the portfolio cannot be processed."""


@dataclass(frozen=True)
class PortfolioHistoryResult:
    historical: pd.DataFrame
    invalid_tickers: list[str]


@dataclass(frozen=True)
class MonteCarloResult:
    simulated_returns: np.ndarray
    var_5: float
    cvar_5: float


@dataclass(frozen=True)
class MarkowitzResult:
    portfolios: pd.DataFrame
    frontier: pd.DataFrame
    max_sharpe_weights: pd.Series
    min_vol_weights: pd.Series
    max_sharpe_return: float
    max_sharpe_volatility: float
    min_vol_return: float
    min_vol_volatility: float


def empty_portfolio_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=DEFAULT_PORTFOLIO_COLUMNS)


def ensure_portfolio_schema(portfolio_df: pd.DataFrame) -> pd.DataFrame:
    working_df = portfolio_df.copy() if not portfolio_df.empty else empty_portfolio_frame()
    for column in DEFAULT_PORTFOLIO_COLUMNS:
        if column not in working_df.columns:
            working_df[column] = pd.NA
    return working_df[DEFAULT_PORTFOLIO_COLUMNS]


def normalize_equity_ticker(ticker_input: str) -> str:
    ticker = ticker_input.strip().upper()
    if re.fullmatch(r"[A-Z]{4}\d{1,2}", ticker) and not ticker.endswith(".SA"):
        return f"{ticker}.SA"
    return ticker


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _extract_close_prices(raw_history: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    if raw_history.empty:
        return pd.DataFrame()

    if isinstance(raw_history.columns, pd.MultiIndex):
        if "Close" in raw_history.columns.get_level_values(0):
            close_history = raw_history.xs("Close", axis=1, level=0)
        elif "Close" in raw_history.columns.get_level_values(-1):
            close_history = raw_history.xs("Close", axis=1, level=-1)
        else:
            return pd.DataFrame()
    else:
        if "Close" in raw_history.columns:
            close_history = raw_history[["Close"]].copy()
            close_history.columns = tickers[:1]
        else:
            close_history = raw_history.copy()
            close_history.columns = tickers[: len(close_history.columns)]

    close_history = close_history.loc[:, ~close_history.columns.duplicated()]
    return close_history.apply(pd.to_numeric, errors="coerce").dropna(how="all")


def _business_date_range(start_date: datetime, end_date: datetime) -> pd.DatetimeIndex:
    return pd.date_range(start=start_date, end=end_date, freq="B")


def _build_cdi_series(index: pd.Index) -> pd.Series:
    if len(index) == 0:
        return pd.Series(dtype=float)

    sorted_index = pd.DatetimeIndex(pd.to_datetime(index, errors="coerce")).dropna().sort_values()
    if len(sorted_index) == 0:
        return pd.Series(dtype=float)

    daily_rate = (1 + ANNUAL_CDI_RATE) ** (1 / TRADING_DAYS_PER_YEAR) - 1
    values = np.empty(len(sorted_index), dtype=float)
    values[0] = 100.0
    for i in range(1, len(sorted_index)):
        values[i] = values[i - 1] * (1 + daily_rate)
    return pd.Series(values, index=sorted_index, name="CDI")


def _analysis_start(end_date: pd.Timestamp, value: float, unit: str) -> pd.Timestamp:
    days = value * (30 if unit == "Mês" else 365)
    return end_date - timedelta(days=days)


def _active_assets(portfolio_df: pd.DataFrame) -> pd.DataFrame:
    if portfolio_df.empty:
        return portfolio_df
    working_df = ensure_portfolio_schema(portfolio_df)
    working_df = working_df[working_df["ticker"].notna()].copy()
    working_df["ticker"] = working_df["ticker"].astype(str).str.strip()
    working_df = working_df[working_df["ticker"] != ""]
    return working_df


def _asset_type(row: pd.Series) -> str:
    asset_type = str(row.get("tipo_ativo", "") or "").strip().lower()
    ticker = str(row.get("ticker", "") or "")
    if asset_type:
        return asset_type
    if re.search(r"\(\d{4}\)$", ticker):
        return "titulo_publico"
    return "acao"


def _available_asset_columns(portfolio_df: pd.DataFrame, historical_df: pd.DataFrame) -> list[str]:
    if portfolio_df.empty or historical_df.empty:
        return []

    tickers = _active_assets(portfolio_df)["ticker"].tolist()
    available = [ticker for ticker in tickers if ticker in historical_df.columns and not historical_df[ticker].dropna().empty]
    return available


def _normalized_weights(portfolio_df: pd.DataFrame, columns: list[str]) -> pd.Series:
    active_assets = _active_assets(portfolio_df)
    if active_assets.empty or not columns:
        return pd.Series(dtype=float)

    weights = pd.to_numeric(active_assets.set_index("ticker")["porcentagem_real"], errors="coerce").fillna(0) / 100
    weights = weights.reindex(columns).fillna(0)
    total = float(weights.sum())
    if total <= 0:
        desired = pd.to_numeric(active_assets.set_index("ticker")["porcentagem_desejada"], errors="coerce").fillna(0) / 100
        weights = desired.reindex(columns).fillna(0)
        total = float(weights.sum())
    if total <= 0:
        return pd.Series(dtype=float)
    return weights / total


def _portfolio_price_series(portfolio_df: pd.DataFrame, historical_df: pd.DataFrame) -> pd.Series:
    columns = _available_asset_columns(portfolio_df, historical_df)
    if not columns:
        return pd.Series(dtype=float)

    weights = _normalized_weights(portfolio_df, columns)
    if weights.empty:
        return pd.Series(dtype=float)

    asset_prices = historical_df[columns].ffill().bfill().dropna(how="all")
    if asset_prices.empty:
        return pd.Series(dtype=float)

    portfolio_prices = asset_prices.multiply(weights, axis=1).sum(axis=1)
    portfolio_prices.name = "Portfolio"
    return portfolio_prices.dropna()


def _portfolio_log_returns(portfolio_df: pd.DataFrame, historical_df: pd.DataFrame) -> pd.Series:
    columns = _available_asset_columns(portfolio_df, historical_df)
    if not columns:
        return pd.Series(dtype=float)

    weights = _normalized_weights(portfolio_df, columns)
    if weights.empty:
        return pd.Series(dtype=float)

    asset_prices = historical_df[columns].ffill().bfill()
    log_returns = np.log(asset_prices / asset_prices.shift(1)).dropna(how="all")
    if log_returns.empty:
        return pd.Series(dtype=float)

    portfolio_returns = log_returns[columns].multiply(weights, axis=1).sum(axis=1)
    portfolio_returns.name = "Portfolio"
    return portfolio_returns.dropna()


@st.cache_resource(show_spinner=False)
def get_treasury_manager() -> "TreasuryDataManager":
    return TreasuryDataManager()


class TreasuryDataManager:
    def __init__(self) -> None:
        self.indicator_history: dict[str, pd.Series] = {}
        self._load_indicator_history()

    def _load_indicator_history(self) -> None:
        if sgs is None:
            return

        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=5 * 365)
        series_map = {"selic": 11, "ipca": 433, "igpm": 189}

        for label, code in series_map.items():
            try:
                series_df = sgs.get({label: code}, start=start_date, end=end_date)
            except Exception:
                continue
            if series_df.empty or label not in series_df.columns:
                continue
            self.indicator_history[label] = pd.to_numeric(series_df[label], errors="coerce").dropna() / 100

    def calculate_theoretical_price_history(
        self,
        title_type: str,
        annual_rate: float,
        current_price: float,
        base_date: Any,
    ) -> pd.Series:
        end_date = pd.to_datetime(base_date, errors="coerce")
        if pd.isna(end_date):
            end_date = pd.Timestamp(datetime.now().date())

        start_date = end_date - timedelta(days=5 * 365)
        date_range = _business_date_range(start_date.to_pydatetime(), end_date.to_pydatetime())
        if len(date_range) == 0:
            return pd.Series(dtype=float)

        annual_rate = _as_float(annual_rate, 6.0)
        current_price = _as_float(current_price, 1000.0)

        indexer = "ipca"
        if "Selic" in title_type:
            indexer = "selic"
        elif "IGP" in title_type:
            indexer = "igpm"
        elif "Prefixado" in title_type:
            indexer = "prefixado"

        if indexer == "selic":
            series = self._stable_selic_series(date_range, current_price)
        elif indexer == "prefixado":
            series = self._stable_prefixed_series(date_range, current_price, annual_rate, end_date)
        else:
            series = self._stable_ipca_series(date_range, current_price, annual_rate)

        if series.empty:
            return pd.Series(dtype=float)

        series = series.interpolate(method="linear").ffill().bfill()
        return series.rolling(window=3, center=True, min_periods=1).mean().dropna()

    def _stable_ipca_series(self, date_range: pd.DatetimeIndex, initial_price: float, real_rate: float) -> pd.Series:
        values = np.empty(len(date_range), dtype=float)
        values[0] = initial_price
        ipca_annual_mean = 0.045
        ipca_volatility = 0.015

        for i in range(1, len(date_range)):
            rng = np.random.default_rng(i)
            ipca_variation = rng.normal(0, ipca_volatility / np.sqrt(TRADING_DAYS_PER_YEAR))
            ipca_daily = (1 + ipca_annual_mean + ipca_variation) ** (1 / TRADING_DAYS_PER_YEAR) - 1
            real_rate_daily = (1 + real_rate / 100) ** (1 / TRADING_DAYS_PER_YEAR) - 1
            values[i] = values[i - 1] * (1 + ipca_daily + real_rate_daily)

        return pd.Series(values, index=date_range)

    def _stable_selic_series(self, date_range: pd.DatetimeIndex, initial_price: float) -> pd.Series:
        daily_rate = (1 + ANNUAL_CDI_RATE) ** (1 / TRADING_DAYS_PER_YEAR) - 1
        growth = np.power(1 + daily_rate, np.arange(len(date_range), dtype=float))
        return pd.Series(initial_price * growth, index=date_range)

    def _stable_prefixed_series(
        self,
        date_range: pd.DatetimeIndex,
        initial_price: float,
        fixed_rate: float,
        maturity_date: pd.Timestamp,
    ) -> pd.Series:
        values = []
        for date in date_range:
            days_to_maturity = max((maturity_date - date).days, 0)
            if days_to_maturity == 0:
                values.append(initial_price)
                continue
            discount_factor = (1 + fixed_rate / 100) ** (days_to_maturity / 365)
            values.append(initial_price / discount_factor)
        return pd.Series(values, index=date_range)


@st.cache_data(ttl=3600, show_spinner=False)
def load_treasury_data() -> pd.DataFrame:
    try:
        treasury_df = pd.read_csv(TREASURY_URL, sep=";", decimal=",")
    except Exception:
        return _fallback_treasury_data()

    treasury_df.columns = (
        treasury_df.columns.str.strip().str.replace(" ", "_").str.replace("ã", "a").str.replace("ç", "c")
    )

    required_columns = {
        "Tipo_Titulo",
        "Data_Vencimento",
        "Data_Base",
        "PU_Compra_Manha",
        "Taxa_Compra_Manha",
    }
    if not required_columns.issubset(set(treasury_df.columns)):
        return _fallback_treasury_data()

    treasury_df["Data_Base"] = pd.to_datetime(treasury_df["Data_Base"], dayfirst=True, errors="coerce")
    treasury_df["Data_Vencimento"] = pd.to_datetime(treasury_df["Data_Vencimento"], dayfirst=True, errors="coerce")
    treasury_df["PU_Compra_Manha"] = pd.to_numeric(treasury_df["PU_Compra_Manha"], errors="coerce")
    treasury_df["Taxa_Compra_Manha"] = pd.to_numeric(treasury_df["Taxa_Compra_Manha"], errors="coerce")
    treasury_df = treasury_df.dropna(subset=["Tipo_Titulo", "Data_Vencimento", "Data_Base", "PU_Compra_Manha"])

    today = pd.Timestamp(datetime.today().date())
    treasury_df = treasury_df[treasury_df["Data_Vencimento"] > today].copy()
    if treasury_df.empty:
        return _fallback_treasury_data()

    latest_base_date = treasury_df["Data_Base"].max()
    return treasury_df[treasury_df["Data_Base"] == latest_base_date].copy()


def _fallback_treasury_data() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for year in [2027, 2029, 2031, 2035, 2040, 2045, 2050]:
        for title_type, label in [("Tesouro IPCA+", "IPCA"), ("Tesouro Selic", "Selic"), ("Tesouro Prefixado", "Prefixado")]:
            rows.append(
                {
                    "Tipo_Titulo": f"{title_type} {label} {year}",
                    "Data_Vencimento": pd.Timestamp(f"{year}-01-01"),
                    "Data_Base": pd.Timestamp(datetime.today().date()),
                    "PU_Compra_Manha": 1000.0 + (year - 2026) * 35,
                    "Taxa_Compra_Manha": 5.5 + (year - 2026) * 0.12,
                }
            )
    return pd.DataFrame(rows)


def treasury_year_options(treasury_df: pd.DataFrame, selected_type: str) -> list[int]:
    if treasury_df.empty:
        return []

    search_term = TREASURY_TYPE_MAP.get(selected_type, selected_type)
    filtered = treasury_df[treasury_df["Tipo_Titulo"].str.contains(search_term, case=False, na=False)]
    if filtered.empty:
        return []
    return sorted(filtered["Data_Vencimento"].dt.year.unique().tolist())


def create_equity_position(
    portfolio_df: pd.DataFrame,
    ticker_input: str,
    target_weight_pct: float,
    total_pl: float,
) -> dict[str, Any]:
    normalized_ticker = normalize_equity_ticker(ticker_input)
    if not normalized_ticker:
        raise PortfolioAnalyticsError("Informe um ticker antes de adicionar o ativo.")

    portfolio_df = ensure_portfolio_schema(portfolio_df)
    if normalized_ticker in portfolio_df["ticker"].astype(str).tolist():
        raise PortfolioAnalyticsError(f"{normalized_ticker} ja esta no portfolio.")

    if target_weight_pct <= 0 or target_weight_pct > 100:
        raise PortfolioAnalyticsError("O peso do ativo precisa estar entre 0 e 100.")

    try:
        asset = yf.Ticker(normalized_ticker)
        info = asset.info or {}
        history = asset.history(period="5d", auto_adjust=False)
    except Exception as exc:
        raise PortfolioAnalyticsError("Falha ao consultar o Yahoo Finance.") from exc

    if history.empty:
        raise PortfolioAnalyticsError(f'O ticker "{normalized_ticker}" nao retornou preco recente.')

    closes = pd.to_numeric(history["Close"], errors="coerce").dropna()
    if closes.empty:
        raise PortfolioAnalyticsError(f'O ticker "{normalized_ticker}" nao retornou fechamento utilizavel.')

    last_price = float(closes.iloc[-1])
    close_date = closes.index[-1].strftime("%d/%m/%Y")
    desired_value = (target_weight_pct / 100) * total_pl
    quantity = int(desired_value / last_price) if last_price > 0 else 0
    real_value = quantity * last_price
    real_weight = (real_value / total_pl) * 100 if total_pl > 0 else 0
    company_name = info.get("shortName") or info.get("longName") or normalized_ticker

    return {
        "ticker": normalized_ticker,
        "nome": company_name,
        "tipo_ativo": "acao",
        "preco": last_price,
        "taxa": 0.0,
        "porcentagem_desejada": target_weight_pct,
        "porcentagem_real": real_weight,
        "qtd_acoes": quantity,
        "valor_real": real_value,
        "data_fechamento": close_date,
        "data_vencimento": pd.NA,
    }


def create_treasury_position(
    portfolio_df: pd.DataFrame,
    treasury_df: pd.DataFrame,
    title_type: str,
    maturity_year: int,
    target_weight_pct: float,
    total_pl: float,
) -> dict[str, Any]:
    if treasury_df.empty:
        raise PortfolioAnalyticsError("Os dados do Tesouro Direto nao estao disponiveis no momento.")

    if target_weight_pct <= 0 or target_weight_pct > 100:
        raise PortfolioAnalyticsError("O peso do titulo precisa estar entre 0 e 100.")

    portfolio_df = ensure_portfolio_schema(portfolio_df)
    ticker_label = f"{title_type} ({maturity_year})"
    if ticker_label in portfolio_df["ticker"].astype(str).tolist():
        raise PortfolioAnalyticsError(f"{ticker_label} ja esta no portfolio.")

    search_term = TREASURY_TYPE_MAP.get(title_type, title_type)
    filtered = treasury_df[
        treasury_df["Tipo_Titulo"].str.contains(search_term, case=False, na=False)
        & (treasury_df["Data_Vencimento"].dt.year == maturity_year)
    ].copy()

    if filtered.empty:
        raise PortfolioAnalyticsError("Titulo nao encontrado nos dados atuais do Tesouro Direto.")

    today = pd.Timestamp(datetime.today().date())
    filtered["date_distance"] = (filtered["Data_Base"] - today).abs().dt.days
    title_row = filtered.sort_values("date_distance").iloc[0]

    purchase_price = _as_float(title_row["PU_Compra_Manha"], 1000.0)
    purchase_rate = _as_float(title_row["Taxa_Compra_Manha"], 0.0)
    maturity_date = pd.to_datetime(title_row["Data_Vencimento"], errors="coerce")
    custody_fee = 0.0020
    days_to_maturity = max((maturity_date - today).days, 0)
    desired_value = (target_weight_pct / 100) * total_pl

    gross_unit_cost = purchase_price * (1 + custody_fee * (days_to_maturity / 365))
    quantity = desired_value / gross_unit_cost if gross_unit_cost > 0 else 0.0
    title_value = quantity * purchase_price
    custody_cost = title_value * custody_fee * (days_to_maturity / 365)
    real_value = title_value + custody_cost
    real_weight = (real_value / total_pl) * 100 if total_pl > 0 else 0

    return {
        "ticker": ticker_label,
        "nome": title_type,
        "tipo_ativo": "titulo_publico",
        "preco": purchase_price,
        "taxa": purchase_rate,
        "porcentagem_desejada": target_weight_pct,
        "porcentagem_real": real_weight,
        "qtd_acoes": round(quantity, 2),
        "valor_real": real_value,
        "data_fechamento": pd.to_datetime(title_row["Data_Base"]).strftime("%d/%m/%Y"),
        "data_vencimento": maturity_date.strftime("%Y-%m-%d") if not pd.isna(maturity_date) else pd.NA,
    }


def recalculate_portfolio(portfolio_df: pd.DataFrame, total_pl: float) -> pd.DataFrame:
    portfolio_df = ensure_portfolio_schema(portfolio_df)
    if portfolio_df.empty or total_pl <= 0:
        return portfolio_df

    working_df = portfolio_df.copy()
    today = pd.Timestamp(datetime.today().date())

    for row_index, row in working_df.iterrows():
        desired_weight = _as_float(row["porcentagem_desejada"])
        price = _as_float(row["preco"])
        desired_value = (desired_weight / 100) * total_pl
        asset_type = _asset_type(row)

        if price <= 0 or desired_value <= 0:
            working_df.at[row_index, "qtd_acoes"] = 0
            working_df.at[row_index, "valor_real"] = 0.0
            working_df.at[row_index, "porcentagem_real"] = 0.0
            continue

        if asset_type == "titulo_publico":
            maturity_date = pd.to_datetime(row.get("data_vencimento"), errors="coerce")
            days_to_maturity = max((maturity_date - today).days, 0) if not pd.isna(maturity_date) else 0
            custody_multiplier = 1 + 0.0020 * (days_to_maturity / 365)
            quantity = desired_value / (price * custody_multiplier)
            real_value = quantity * price * custody_multiplier
            working_df.at[row_index, "qtd_acoes"] = round(quantity, 2)
        else:
            quantity = int(desired_value / price)
            real_value = quantity * price
            working_df.at[row_index, "qtd_acoes"] = int(quantity)

        working_df.at[row_index, "valor_real"] = real_value
        working_df.at[row_index, "porcentagem_real"] = (real_value / total_pl) * 100

    return ensure_portfolio_schema(working_df)


def append_position(portfolio_df: pd.DataFrame, position: dict[str, Any], total_pl: float) -> pd.DataFrame:
    portfolio_df = ensure_portfolio_schema(portfolio_df)
    updated = pd.concat([portfolio_df, pd.DataFrame([position])], ignore_index=True)
    return recalculate_portfolio(updated, total_pl)


def remove_position(portfolio_df: pd.DataFrame, ticker: str, total_pl: float) -> pd.DataFrame:
    portfolio_df = ensure_portfolio_schema(portfolio_df)
    updated = portfolio_df[portfolio_df["ticker"].astype(str) != ticker].copy()
    return recalculate_portfolio(updated, total_pl)


def build_historical_dataset(portfolio_df: pd.DataFrame) -> PortfolioHistoryResult:
    active_assets = _active_assets(portfolio_df)
    if active_assets.empty:
        return PortfolioHistoryResult(historical=pd.DataFrame(), invalid_tickers=[])

    start_date = datetime.now() - timedelta(days=5 * 365 + 180)
    end_date = datetime.now()
    historical_df = pd.DataFrame()
    invalid_tickers: list[str] = []

    equity_assets = active_assets[active_assets.apply(_asset_type, axis=1) != "titulo_publico"].copy()
    treasury_assets = active_assets[active_assets.apply(_asset_type, axis=1) == "titulo_publico"].copy()

    if not equity_assets.empty:
        equity_tickers = equity_assets["ticker"].astype(str).drop_duplicates().tolist()
        requested = equity_tickers + ["^BVSP"]
        try:
            raw_history = yf.download(
                tickers=requested,
                start=start_date,
                end=end_date,
                auto_adjust=True,
                progress=False,
                threads=False,
            )
        except Exception as exc:
            raise PortfolioAnalyticsError("Falha ao baixar o historico do Yahoo Finance.") from exc

        close_history = _extract_close_prices(raw_history, requested)
        if not close_history.empty:
            close_history = close_history.rename(columns={"^BVSP": "IBOVESPA"})
            historical_df = close_history.copy()
            invalid_tickers.extend([ticker for ticker in equity_tickers if ticker not in close_history.columns])
        else:
            invalid_tickers.extend(equity_tickers)

    if not treasury_assets.empty:
        treasury_manager = get_treasury_manager()
        for _, row in treasury_assets.iterrows():
            ticker = str(row["ticker"])
            title_name = str(row.get("nome") or ticker)
            annual_rate = _as_float(row.get("taxa"), 0.0)
            current_price = _as_float(row.get("preco"), 1000.0)
            base_date = row.get("data_fechamento") or datetime.now().strftime("%d/%m/%Y")

            history = treasury_manager.calculate_theoretical_price_history(
                title_name,
                annual_rate,
                current_price,
                base_date,
            )
            if history.empty:
                continue

            if historical_df.empty:
                historical_df = pd.DataFrame(index=history.index)
            historical_df = historical_df.reindex(historical_df.index.union(history.index))
            historical_df[ticker] = history

    if not historical_df.empty and "IBOVESPA" not in historical_df.columns:
        try:
            ibov_history = yf.download("^BVSP", start=start_date, end=end_date, progress=False, auto_adjust=True, threads=False)
            close_prices = _extract_close_prices(ibov_history, ["^BVSP"])
            if not close_prices.empty and "^BVSP" in close_prices.columns:
                historical_df = historical_df.reindex(historical_df.index.union(close_prices.index))
                historical_df["IBOVESPA"] = close_prices["^BVSP"]
        except Exception:
            pass

    if historical_df.empty:
        return PortfolioHistoryResult(historical=pd.DataFrame(), invalid_tickers=sorted(set(invalid_tickers)))

    historical_df.index = pd.to_datetime(historical_df.index, errors="coerce")
    historical_df = historical_df[historical_df.index.notna()].sort_index()
    historical_df = historical_df.apply(pd.to_numeric, errors="coerce")
    historical_df = historical_df.dropna(how="all")
    historical_df = historical_df.ffill()
    historical_df["CDI"] = _build_cdi_series(historical_df.index)
    historical_df = historical_df.loc[:, ~historical_df.columns.duplicated()]
    return PortfolioHistoryResult(historical=historical_df, invalid_tickers=sorted(set(invalid_tickers)))


def portfolio_overview(portfolio_df: pd.DataFrame, historical_df: pd.DataFrame, total_pl: float) -> dict[str, Any]:
    active_assets = _active_assets(portfolio_df)
    total_invested = float(pd.to_numeric(active_assets["valor_real"], errors="coerce").fillna(0).sum()) if not active_assets.empty else 0.0
    return {
        "asset_count": int(len(active_assets)),
        "invested_value": total_invested,
        "cash_remaining": max(total_pl - total_invested, 0.0),
        "latest_history_date": historical_df.index.max().strftime("%d/%m/%Y") if not historical_df.empty else "N/A",
    }


def normalized_history_with_portfolio(portfolio_df: pd.DataFrame, historical_df: pd.DataFrame) -> pd.DataFrame:
    price_series = _portfolio_price_series(portfolio_df, historical_df)
    columns = _available_asset_columns(portfolio_df, historical_df)
    if price_series.empty or not columns:
        return pd.DataFrame()

    asset_history = historical_df[columns].ffill().bfill().dropna(how="all")
    base_prices = asset_history.bfill().iloc[0].replace(0, pd.NA)
    normalized_assets = asset_history.divide(base_prices).multiply(100)
    normalized_portfolio = price_series.divide(price_series.iloc[0]).multiply(100)
    normalized_assets["Portfolio"] = normalized_portfolio
    return normalized_assets.dropna(how="all")


def benchmark_return_series(
    portfolio_df: pd.DataFrame,
    historical_df: pd.DataFrame,
    benchmark_column: str,
    *,
    years: int | None = None,
    months: int | None = None,
) -> pd.DataFrame:
    if historical_df.empty or benchmark_column not in historical_df.columns:
        return pd.DataFrame()

    portfolio_prices = _portfolio_price_series(portfolio_df, historical_df)
    if portfolio_prices.empty:
        return pd.DataFrame()

    end_date = historical_df.index.max()
    start_date = end_date
    if months is not None:
        start_date = end_date - pd.DateOffset(months=months)
    if years is not None:
        start_date = end_date - pd.DateOffset(years=years)

    period_df = pd.concat(
        [portfolio_prices.rename("Portfolio"), historical_df[benchmark_column].rename(benchmark_column)],
        axis=1,
    ).loc[start_date:end_date]
    period_df = period_df.ffill().bfill().dropna()
    if len(period_df) < 2:
        return pd.DataFrame()

    base_values = period_df.iloc[0].replace(0, pd.NA)
    return period_df.divide(base_values).subtract(1).multiply(100)


def contribution_by_asset(portfolio_df: pd.DataFrame, historical_df: pd.DataFrame) -> pd.DataFrame:
    if historical_df.empty:
        return pd.DataFrame()

    active_assets = _active_assets(portfolio_df)
    if active_assets.empty:
        return pd.DataFrame()

    end_date = historical_df.index.max()
    start_date = end_date - pd.DateOffset(years=1)
    period_df = historical_df.loc[start_date:end_date]
    rows: list[dict[str, Any]] = []

    for _, row in active_assets.iterrows():
        ticker = str(row["ticker"])
        if ticker not in period_df.columns:
            continue
        series = period_df[ticker].ffill().bfill().dropna()
        if len(series) < 2:
            continue
        asset_return = (series.iloc[-1] / series.iloc[0]) - 1
        weight = _as_float(row["porcentagem_real"]) / 100
        rows.append({"ticker": ticker, "contribuicao": asset_return * weight})

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("contribuicao", ascending=False)


def drawdown_series(portfolio_df: pd.DataFrame, historical_df: pd.DataFrame) -> pd.Series:
    portfolio_prices = _portfolio_price_series(portfolio_df, historical_df)
    if len(portfolio_prices) < 2:
        return pd.Series(dtype=float)
    peak = portfolio_prices.cummax()
    drawdown = ((portfolio_prices - peak) / peak) * 100
    drawdown.name = "Drawdown"
    return drawdown.dropna()


def monte_carlo_simulation(portfolio_df: pd.DataFrame, historical_df: pd.DataFrame) -> MonteCarloResult | None:
    columns = _available_asset_columns(portfolio_df, historical_df)
    if len(columns) == 0:
        return None

    weights = _normalized_weights(portfolio_df, columns)
    prices = historical_df[columns].ffill().bfill()
    log_returns = np.log(prices / prices.shift(1)).dropna()
    if len(log_returns) < 20:
        return None

    portfolio_log_returns = log_returns[columns].multiply(weights, axis=1).sum(axis=1).dropna()
    if portfolio_log_returns.empty:
        return None

    rng = np.random.default_rng(42)
    simulations = rng.choice(portfolio_log_returns.values, size=(20, 10000), replace=True)
    results = np.exp(simulations.sum(axis=0)) - 1
    threshold = np.percentile(results, 5)
    cvar = results[results <= threshold].mean() if np.any(results <= threshold) else threshold
    return MonteCarloResult(simulated_returns=results, var_5=threshold * 100, cvar_5=cvar * 100)


def rolling_volatility_series(portfolio_df: pd.DataFrame, historical_df: pd.DataFrame) -> pd.Series:
    portfolio_prices = _portfolio_price_series(portfolio_df, historical_df)
    if len(portfolio_prices) < 21:
        return pd.Series(dtype=float)
    returns = portfolio_prices.pct_change().dropna()
    rolling_vol = returns.rolling(window=21).std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    rolling_vol.name = "Volatilidade"
    return rolling_vol.dropna()


def individual_metrics(portfolio_df: pd.DataFrame, historical_df: pd.DataFrame, value: float, unit: str) -> dict[str, dict[str, float]] | None:
    if historical_df.empty:
        return None

    end_date = historical_df.index.max()
    start_date = _analysis_start(end_date, value, unit)
    period_df = historical_df.loc[start_date:end_date].ffill().bfill()
    if period_df.empty:
        return None

    log_returns = np.log(period_df / period_df.shift(1)).dropna(how="all")
    active_assets = _active_assets(portfolio_df)
    valid_tickers = [ticker for ticker in active_assets["ticker"].tolist() if ticker in log_returns.columns]
    if not valid_tickers:
        return None

    metrics: dict[str, dict[str, float]] = {}
    for ticker in valid_tickers:
        asset_returns = log_returns[ticker].dropna()
        if asset_returns.empty:
            continue
        metrics[ticker] = {
            "media": float(asset_returns.mean()),
            "variancia": float(asset_returns.var()),
            "volatilidade": float(asset_returns.std()),
        }

    portfolio_returns = _portfolio_log_returns(portfolio_df, period_df)
    if not portfolio_returns.empty:
        metrics["Portfolio"] = {
            "media": float(portfolio_returns.mean()),
            "variancia": float(portfolio_returns.var()),
            "volatilidade": float(portfolio_returns.std()),
        }

    return metrics or None


def var_cvar_metrics(
    portfolio_df: pd.DataFrame,
    historical_df: pd.DataFrame,
    total_pl: float,
    value: float,
    unit: str,
    confidence: float,
) -> dict[str, dict[str, float]] | None:
    metrics = individual_metrics(portfolio_df, historical_df, value, unit)
    if not metrics:
        return None

    z_score = norm.ppf(confidence)
    active_assets = _active_assets(portfolio_df).set_index("ticker")
    result: dict[str, dict[str, float]] = {}

    for asset, stats in metrics.items():
        var_pct = -(stats["media"] + z_score * stats["volatilidade"])
        cvar_pct = (1 / (1 - confidence)) * norm.pdf(z_score) * stats["volatilidade"] - stats["media"]
        invested_value = total_pl if asset == "Portfolio" else _as_float(active_assets["valor_real"].get(asset), 0.0)
        result[asset] = {"var": var_pct * invested_value, "cvar": cvar_pct * invested_value}

    return result


def capm_alpha_beta_correlation(
    portfolio_df: pd.DataFrame,
    historical_df: pd.DataFrame,
    value: float,
    unit: str,
) -> dict[str, float] | None:
    if historical_df.empty or "IBOVESPA" not in historical_df.columns:
        return None

    end_date = historical_df.index.max()
    start_date = _analysis_start(end_date, value, unit)
    period_df = historical_df.loc[start_date:end_date].ffill().bfill()
    if period_df.empty:
        return None

    log_returns = np.log(period_df / period_df.shift(1))
    portfolio_returns = _portfolio_log_returns(portfolio_df, period_df)
    if portfolio_returns.empty or "IBOVESPA" not in log_returns.columns:
        return None

    combined = pd.concat([portfolio_returns, log_returns["IBOVESPA"]], axis=1, keys=["portfolio", "ibov"]).dropna()
    if len(combined) < 2:
        return None

    ibov_returns = combined["ibov"]
    cdi_daily = (1 + ANNUAL_CDI_RATE) ** (1 / TRADING_DAYS_PER_YEAR) - 1

    try:
        model = sm.OLS(combined["portfolio"], sm.add_constant(ibov_returns)).fit()
        alpha, beta = model.params
    except Exception:
        alpha, beta = 0.0, 1.0

    capm_daily = cdi_daily + beta * (ibov_returns.mean() - cdi_daily)
    return {
        "capm": float((1 + capm_daily) ** TRADING_DAYS_PER_YEAR - 1),
        "alfa": float(alpha),
        "beta": float(beta),
        "correlacao": float(combined["portfolio"].corr(ibov_returns)),
    }


def performance_indices(
    portfolio_df: pd.DataFrame,
    historical_df: pd.DataFrame,
    value: float,
    unit: str,
) -> dict[str, dict[str, float]] | None:
    if historical_df.empty or "IBOVESPA" not in historical_df.columns:
        return None

    end_date = historical_df.index.max()
    start_date = _analysis_start(end_date, value, unit)
    period_df = historical_df.loc[start_date:end_date].ffill().bfill()
    if period_df.empty:
        return None

    log_returns = np.log(period_df / period_df.shift(1))
    portfolio_returns = _portfolio_log_returns(portfolio_df, period_df)
    combined = pd.concat([portfolio_returns, log_returns["IBOVESPA"]], axis=1, keys=["portfolio", "ibov"]).dropna()
    if len(combined) < 2:
        return None

    portfolio_ret = combined["portfolio"]
    ibov_ret = combined["ibov"]
    cdi_daily = (1 + ANNUAL_CDI_RATE) ** (1 / TRADING_DAYS_PER_YEAR) - 1

    mean_portfolio = portfolio_ret.mean()
    portfolio_volatility = portfolio_ret.std()
    downside_returns = portfolio_ret[portfolio_ret < 0]
    downside_deviation = downside_returns.std(ddof=0) if len(downside_returns) > 0 else 0.0

    try:
        model = sm.OLS(portfolio_ret, sm.add_constant(ibov_ret)).fit()
        _, beta = model.params
    except Exception:
        beta = 1.0

    tracking_error = (portfolio_ret - ibov_ret).std()
    annual_return = (1 + mean_portfolio) ** TRADING_DAYS_PER_YEAR - 1
    annual_volatility = portfolio_volatility * np.sqrt(TRADING_DAYS_PER_YEAR)
    annual_downside = downside_deviation * np.sqrt(TRADING_DAYS_PER_YEAR) if downside_deviation > 0 else 0.0
    annual_tracking_error = tracking_error * np.sqrt(TRADING_DAYS_PER_YEAR)

    return {
        "diario": {
            "sharpe": float((mean_portfolio - cdi_daily) / portfolio_volatility) if portfolio_volatility > 0 else 0.0,
            "sortino": float((mean_portfolio - cdi_daily) / downside_deviation) if downside_deviation > 0 else 0.0,
            "treynor": float((mean_portfolio - cdi_daily) / beta) if beta != 0 else 0.0,
            "tracking_error": float(tracking_error),
        },
        "anual": {
            "sharpe": float((annual_return - ANNUAL_CDI_RATE) / annual_volatility) if annual_volatility > 0 else 0.0,
            "sortino": float((annual_return - ANNUAL_CDI_RATE) / annual_downside) if annual_downside > 0 else 0.0,
            "treynor": float((annual_return - ANNUAL_CDI_RATE) / beta) if beta != 0 else 0.0,
            "tracking_error": float(annual_tracking_error),
        },
    }


def correlation_matrix(portfolio_df: pd.DataFrame, historical_df: pd.DataFrame, value: float, unit: str) -> tuple[pd.DataFrame, list[str]] | None:
    if historical_df.empty:
        return None

    end_date = historical_df.index.max()
    start_date = _analysis_start(end_date, value, unit)
    period_df = historical_df.loc[start_date:end_date].ffill().bfill()
    log_returns = np.log(period_df / period_df.shift(1)).dropna(how="all")
    if log_returns.empty:
        return None

    active_assets = _active_assets(portfolio_df)
    valid_tickers = [ticker for ticker in active_assets["ticker"].tolist() if ticker in log_returns.columns]
    if len(valid_tickers) < 2:
        return None
    return log_returns[valid_tickers].corr(), valid_tickers


def accumulated_returns_table(
    portfolio_df: pd.DataFrame,
    historical_df: pd.DataFrame,
    value: float,
    unit: str,
) -> tuple[dict[int, dict[int, float]], dict[int, float], float, pd.Timestamp, pd.Timestamp] | None:
    if historical_df.empty:
        return None

    end_date = historical_df.index.max()
    start_period = _analysis_start(end_date, value, unit)
    start_table = end_date - timedelta(days=5 * 365)
    period_df = historical_df.loc[start_table:end_date].ffill().bfill()
    if period_df.empty:
        return None

    portfolio_returns = _portfolio_log_returns(portfolio_df, period_df)
    if portfolio_returns.empty:
        return None

    monthly_returns = portfolio_returns.resample("ME").apply(lambda values: np.exp(values.sum()) - 1)
    yearly_returns = portfolio_returns.resample("YE").apply(lambda values: np.exp(values.sum()) - 1)
    period_return = float(np.exp(portfolio_returns.loc[start_period:end_date].sum()) - 1)

    monthly_table: dict[int, dict[int, float]] = {}
    for date, value_ in monthly_returns.items():
        monthly_table.setdefault(date.year, {})[date.month] = float(value_)

    yearly_table = {date.year: float(value_) for date, value_ in yearly_returns.items()}
    return monthly_table, yearly_table, period_return, start_period, end_date


def markowitz_frontier(portfolio_df: pd.DataFrame, historical_df: pd.DataFrame) -> MarkowitzResult | None:
    columns = _available_asset_columns(portfolio_df, historical_df)
    if len(columns) < 2:
        return None

    prices = historical_df[columns].ffill().bfill()
    returns = prices.pct_change().dropna()
    if len(returns) < 21:
        return None

    mean_returns = returns.mean()
    covariance = returns.cov()
    eigenvalues = np.linalg.eigvals(covariance.values)
    if np.any(eigenvalues <= 0):
        covariance = covariance + np.eye(len(columns)) * 1e-8

    rng = np.random.default_rng(42)
    results = np.zeros((3 + len(columns), 10000))

    for index in range(10000):
        weights = rng.random(len(columns))
        weights = weights / weights.sum()
        annual_return = float(np.dot(weights, mean_returns) * TRADING_DAYS_PER_YEAR)
        annual_volatility = float(np.sqrt(np.dot(weights.T, np.dot(covariance, weights))) * np.sqrt(TRADING_DAYS_PER_YEAR))
        sharpe = annual_return / annual_volatility if annual_volatility > 0 else 0.0
        results[0, index] = annual_return
        results[1, index] = annual_volatility
        results[2, index] = sharpe
        results[3 :, index] = weights

    portfolios = pd.DataFrame(results.T, columns=["retorno", "volatilidade", "sharpe", *columns])
    max_sharpe_row = portfolios.loc[portfolios["sharpe"].idxmax()]
    min_vol_row = portfolios.loc[portfolios["volatilidade"].idxmin()]

    frontier_returns = np.linspace(portfolios["retorno"].min(), portfolios["retorno"].max(), 50)
    frontier_vols = []
    for target_return in frontier_returns:
        subset = portfolios[portfolios["retorno"] >= target_return]
        frontier_vols.append(subset["volatilidade"].min() if not subset.empty else np.nan)

    frontier = pd.DataFrame({"volatilidade": frontier_vols, "retorno": frontier_returns})
    return MarkowitzResult(
        portfolios=portfolios,
        frontier=frontier.dropna(),
        max_sharpe_weights=max_sharpe_row[columns],
        min_vol_weights=min_vol_row[columns],
        max_sharpe_return=float(max_sharpe_row["retorno"]),
        max_sharpe_volatility=float(max_sharpe_row["volatilidade"]),
        min_vol_return=float(min_vol_row["retorno"]),
        min_vol_volatility=float(min_vol_row["volatilidade"]),
    )
