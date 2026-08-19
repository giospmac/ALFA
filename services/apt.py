"""APT — Arbitrage Pricing Theory multifactor model.

Fetches macroeconomic factors (IBOV, IPCA, PTAX, Selic), builds a factor
matrix, runs OLS regressions per asset, and computes factor contributions.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

try:
    from bcb import sgs
except ImportError:
    sgs = None


# ------------------------------------------------------------------
# Data classes
# ------------------------------------------------------------------

@dataclass(frozen=True)
class APTRegressionResult:
    ticker: str
    intercept: float
    betas: dict[str, float]
    p_values: dict[str, float]
    r_squared: float
    r_squared_adj: float
    f_stat: float
    f_pvalue: float
    residuals: pd.Series


@dataclass(frozen=True)
class APTFullResult:
    regressions: list[APTRegressionResult]
    vif: pd.DataFrame | None
    factor_names: list[str]


# ------------------------------------------------------------------
# Factor pipeline
# ------------------------------------------------------------------

def fetch_factor_series(
    start_date: datetime | None = None,
    end_date: datetime | None = None,
) -> dict[str, pd.Series]:
    """Fetch macro factors from BCB SGS.

    Returns dict with keys: ibov_ret, ipca, ptax, selic_var.
    """
    if sgs is None:
        return {}

    if end_date is None:
        end_date = datetime.now()
    if start_date is None:
        start_date = end_date - timedelta(days=10 * 365)

    series_map = {
        "ipca": 433,
        "selic": 11,
    }

    result: dict[str, pd.Series] = {}

    for label, code in series_map.items():
        try:
            df = sgs.get({label: code}, start=start_date.date(), end=end_date.date())
            if not df.empty and label in df.columns:
                result[label] = pd.to_numeric(df[label], errors="coerce").dropna()
        except Exception:
            continue

    # PTAX (USD/BRL) — code 1 is PTAX sell rate
    try:
        df = sgs.get({"ptax": 1}, start=start_date.date(), end=end_date.date())
        if not df.empty and "ptax" in df.columns:
            result["ptax"] = pd.to_numeric(df["ptax"], errors="coerce").dropna()
    except Exception:
        pass

    return result


def standardize_frequency(
    series_dict: dict[str, pd.Series],
    freq: str = "M",
) -> dict[str, pd.Series]:
    """Resample all series to the given frequency."""
    result = {}
    for name, series in series_dict.items():
        series = series.copy()
        series.index = pd.to_datetime(series.index, errors="coerce")
        series = series[series.index.notna()].sort_index()
        if series.empty:
            continue
        if freq == "M":
            result[name] = series.resample("ME").last().dropna()
        else:
            result[name] = series.resample(freq).last().dropna()
    return result


def transform_factors(
    series_dict: dict[str, pd.Series],
) -> dict[str, pd.Series]:
    """Transform factors into returns / variations as appropriate.

    - IPCA: already monthly % → use as-is (divide by 100)
    - Selic: level → monthly variation
    - PTAX: level → monthly % change
    """
    result = {}

    if "ipca" in series_dict:
        result["Inflação (IPCA)"] = series_dict["ipca"] / 100

    if "selic" in series_dict:
        selic = series_dict["selic"]
        result["ΔSelic"] = selic.diff().dropna() / 100

    if "ptax" in series_dict:
        ptax = series_dict["ptax"]
        result["Câmbio (PTAX)"] = ptax.pct_change().dropna()

    return result


def build_factor_matrix(
    asset_prices: pd.DataFrame,
    factor_series: dict[str, pd.Series],
    freq: str = "M",
    lags: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build aligned factor matrix and asset return matrix.

    Returns (asset_returns, factor_matrix) — both monthly.
    """
    # Asset returns → monthly
    asset_prices.index = pd.to_datetime(asset_prices.index, errors="coerce")
    monthly_prices = asset_prices.resample("ME").last().dropna(how="all")
    asset_returns = monthly_prices.pct_change().dropna(how="all")

    # Market factor (IBOV) from asset_prices if available
    if "IBOVESPA" in asset_prices.columns:
        ibov_monthly = asset_prices["IBOVESPA"].resample("ME").last().dropna()
        ibov_ret = ibov_monthly.pct_change().dropna()
        factor_series["Mercado (IBOV)"] = ibov_ret

    # Build factor DataFrame
    factor_df = pd.DataFrame(factor_series)
    if factor_df.empty:
        return asset_returns, pd.DataFrame()

    # Align on common dates
    common_idx = asset_returns.index.intersection(factor_df.index)
    if len(common_idx) < 12:
        # Try close alignment (within 3 days)
        pass

    asset_returns = asset_returns.reindex(common_idx).dropna(how="all")
    factor_df = factor_df.reindex(common_idx).dropna(how="all")

    # Apply lags if requested
    if lags > 0:
        lagged_cols = {}
        for col in factor_df.columns:
            for lag in range(1, lags + 1):
                lagged_cols[f"{col} (t-{lag})"] = factor_df[col].shift(lag)
        factor_df = pd.concat([factor_df, pd.DataFrame(lagged_cols)], axis=1)
        factor_df = factor_df.dropna()
        asset_returns = asset_returns.reindex(factor_df.index).dropna(how="all")

    return asset_returns, factor_df


# ------------------------------------------------------------------
# Regression
# ------------------------------------------------------------------

def run_apt_regression(
    asset_returns: pd.Series,
    factor_matrix: pd.DataFrame,
) -> APTRegressionResult | None:
    """Run OLS regression of asset returns on factors."""
    combined = pd.concat([asset_returns.rename("y"), factor_matrix], axis=1).dropna()
    if len(combined) < max(10, factor_matrix.shape[1] + 3):
        return None

    y = combined["y"]
    X = sm.add_constant(combined.drop(columns=["y"]))

    try:
        model = sm.OLS(y, X).fit()
    except Exception:
        return None

    factor_names = list(factor_matrix.columns)
    betas = {name: float(model.params.get(name, 0)) for name in factor_names}
    pvals = {name: float(model.pvalues.get(name, 1)) for name in factor_names}

    return APTRegressionResult(
        ticker=asset_returns.name or "",
        intercept=float(model.params.get("const", 0)),
        betas=betas,
        p_values=pvals,
        r_squared=float(model.rsquared),
        r_squared_adj=float(model.rsquared_adj),
        f_stat=float(model.fvalue) if np.isfinite(model.fvalue) else 0.0,
        f_pvalue=float(model.f_pvalue) if np.isfinite(model.f_pvalue) else 1.0,
        residuals=pd.Series(model.resid, index=combined.index),
    )


def compute_vif(factor_matrix: pd.DataFrame) -> pd.DataFrame | None:
    """Variance Inflation Factor for each factor."""
    X = factor_matrix.dropna()
    if X.empty or X.shape[1] < 2 or X.shape[0] < X.shape[1] + 2:
        return None

    X_const = sm.add_constant(X)
    try:
        vif_data = []
        for i in range(1, X_const.shape[1]):
            vif_data.append({
                "Fator": X_const.columns[i],
                "VIF": float(variance_inflation_factor(X_const.values, i)),
            })
        return pd.DataFrame(vif_data)
    except Exception:
        return None


def factor_contribution(
    betas: dict[str, float],
    factor_means: dict[str, float],
) -> dict[str, float]:
    """Expected contribution of each factor to the asset's return."""
    return {name: betas.get(name, 0) * factor_means.get(name, 0) for name in betas}


# ------------------------------------------------------------------
# High-level runner
# ------------------------------------------------------------------

def run_apt_analysis(
    asset_prices: pd.DataFrame,
    tickers: list[str],
    start_date: datetime | None = None,
    end_date: datetime | None = None,
    lags: int = 0,
    enabled_factors: list[str] | None = None,
) -> APTFullResult | None:
    """Full APT pipeline: fetch factors → build matrix → regress each asset."""
    raw_factors = fetch_factor_series(start_date, end_date)
    if not raw_factors and "IBOVESPA" not in asset_prices.columns:
        return None

    monthly_factors = standardize_frequency(raw_factors, "M")
    transformed = transform_factors(monthly_factors)

    asset_returns, factor_matrix = build_factor_matrix(
        asset_prices[tickers + (["IBOVESPA"] if "IBOVESPA" in asset_prices.columns else [])],
        transformed,
        lags=lags,
    )

    if factor_matrix.empty:
        return None

    # Filter to enabled factors
    if enabled_factors:
        cols_to_keep = [c for c in factor_matrix.columns if any(ef in c for ef in enabled_factors)]
        if cols_to_keep:
            factor_matrix = factor_matrix[cols_to_keep]

    # VIF
    vif_df = compute_vif(factor_matrix)

    # Run per-asset regressions
    results: list[APTRegressionResult] = []
    for ticker in tickers:
        if ticker not in asset_returns.columns or ticker == "IBOVESPA":
            continue
        reg = run_apt_regression(asset_returns[ticker].rename(ticker), factor_matrix)
        if reg is not None:
            results.append(reg)

    if not results:
        return None

    return APTFullResult(
        regressions=results,
        vif=vif_df,
        factor_names=list(factor_matrix.columns),
    )
