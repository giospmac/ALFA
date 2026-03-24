"""Forward-looking CAPM, Jensen's Alpha, and Systematic / Idiosyncratic risk decomposition.

All functions are pure (no Streamlit dependency) so they can be unit-tested
and reused across pages.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm

TRADING_DAYS_PER_YEAR = 252


# ------------------------------------------------------------------
# Core CAPM helpers
# ------------------------------------------------------------------

def estimate_beta(asset_returns: pd.Series, market_returns: pd.Series) -> float:
    """OLS beta of *asset_returns* against *market_returns* (daily log-returns)."""
    combined = pd.concat([asset_returns.rename("asset"), market_returns.rename("market")], axis=1).dropna()
    if len(combined) < 10:
        return 1.0
    market_var = float(combined["market"].var())
    if market_var <= 0:
        return 1.0
    return float(combined["asset"].cov(combined["market"]) / market_var)


def capm_expected_return(rf_annual: float, beta: float, emrp_annual: float) -> float:
    """E(Ri) = Rf + β × EMRP  (annualised)."""
    return rf_annual + beta * emrp_annual


def realized_market_premium(market_returns: pd.Series, rf_daily: float) -> float:
    """Annualised realised market premium  (Rm – Rf) from daily log-returns."""
    if market_returns.empty:
        return 0.0
    mean_daily = float(market_returns.mean())
    annual_market = (1 + mean_daily) ** TRADING_DAYS_PER_YEAR - 1
    rf_annual = (1 + rf_daily) ** TRADING_DAYS_PER_YEAR - 1
    return annual_market - rf_annual


def jensen_alpha(
    asset_returns: pd.Series,
    market_returns: pd.Series,
    rf_daily: float,
) -> float:
    """Jensen's alpha (daily) from CAPM regression."""
    combined = pd.concat(
        [asset_returns.rename("asset"), market_returns.rename("market")],
        axis=1,
    ).dropna()
    if len(combined) < 10:
        return 0.0
    y = combined["asset"] - rf_daily
    X = sm.add_constant(combined["market"] - rf_daily)
    try:
        model = sm.OLS(y, X).fit()
        return float(model.params.iloc[0])
    except Exception:
        return 0.0


# ------------------------------------------------------------------
# Risk decomposition
# ------------------------------------------------------------------

@dataclass(frozen=True)
class RiskDecomposition:
    total_var: float
    systematic_var: float
    idiosyncratic_var: float
    total_vol_annual: float
    systematic_vol_annual: float
    idiosyncratic_vol_annual: float
    r_squared: float


def systematic_idiosyncratic_risk(
    asset_returns: pd.Series,
    market_returns: pd.Series,
    beta: float,
) -> RiskDecomposition:
    """Decompose asset variance into systematic (β²σ²_m) and idiosyncratic."""
    combined = pd.concat(
        [asset_returns.rename("asset"), market_returns.rename("market")],
        axis=1,
    ).dropna()
    if len(combined) < 10:
        return RiskDecomposition(0, 0, 0, 0, 0, 0, 0)

    total_var = float(combined["asset"].var())
    market_var = float(combined["market"].var())
    systematic_var = beta ** 2 * market_var
    idiosyncratic_var = max(total_var - systematic_var, 0.0)
    r_squared = systematic_var / total_var if total_var > 0 else 0.0

    sqrt_ann = np.sqrt(TRADING_DAYS_PER_YEAR)
    return RiskDecomposition(
        total_var=total_var,
        systematic_var=systematic_var,
        idiosyncratic_var=idiosyncratic_var,
        total_vol_annual=float(np.sqrt(total_var) * sqrt_ann),
        systematic_vol_annual=float(np.sqrt(systematic_var) * sqrt_ann),
        idiosyncratic_vol_annual=float(np.sqrt(idiosyncratic_var) * sqrt_ann),
        r_squared=float(np.clip(r_squared, 0, 1)),
    )


# ------------------------------------------------------------------
# Per-asset CAPM table builder
# ------------------------------------------------------------------

def capm_per_asset_table(
    portfolio_df: pd.DataFrame,
    historical_df: pd.DataFrame,
    rf_annual: float,
    emrp_annual: float,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DataFrame | None:
    """Build a DataFrame with full CAPM metrics for every portfolio asset.

    Columns: Ativo, Beta, Retorno Histórico, CAPM Esperado, Prêmio Realizado,
    Prêmio Esperado, Vol Total, Vol Sistêmica, Vol Idiossincrática, R²
    """
    if historical_df.empty or "IBOVESPA" not in historical_df.columns:
        return None

    period_df = historical_df.loc[start_date:end_date].ffill().bfill()
    if period_df.empty or "IBOVESPA" not in period_df.columns:
        return None

    log_returns = np.log(period_df / period_df.shift(1)).dropna(how="all")
    market_ret = log_returns["IBOVESPA"].dropna()
    if market_ret.empty:
        return None

    rf_daily = (1 + rf_annual) ** (1 / TRADING_DAYS_PER_YEAR) - 1

    # Identify portfolio tickers (exclude benchmarks)
    from services.portfolio_analytics import _active_assets  # noqa: local import to avoid circular
    active = _active_assets(portfolio_df)
    if active.empty:
        return None

    tickers = [t for t in active["ticker"].tolist() if t in log_returns.columns]
    if not tickers:
        return None

    rows: list[dict] = []
    for ticker in tickers:
        asset_ret = log_returns[ticker].dropna()
        if len(asset_ret) < 20:
            continue

        beta_val = estimate_beta(asset_ret, market_ret)
        hist_return_annual = float(np.exp(asset_ret.sum()) - 1)  # cumulative in period
        n_days = len(asset_ret)
        if n_days > 0:
            hist_return_annual = float((1 + asset_ret.mean()) ** TRADING_DAYS_PER_YEAR - 1)

        capm_er = capm_expected_return(rf_annual, beta_val, emrp_annual)
        realized_premium = realized_market_premium(market_ret, rf_daily)
        expected_premium = emrp_annual
        alpha_val = jensen_alpha(asset_ret, market_ret, rf_daily)

        risk = systematic_idiosyncratic_risk(asset_ret, market_ret, beta_val)

        rows.append({
            "Ativo": ticker.replace(".SA", ""),
            "Beta": round(beta_val, 4),
            "Retorno Histórico": hist_return_annual,
            "CAPM Esperado": capm_er,
            "Alfa de Jensen": alpha_val,
            "Prêmio Realizado": realized_premium,
            "Prêmio Esperado": expected_premium,
            "Vol Total": risk.total_vol_annual,
            "Vol Sistêmica": risk.systematic_vol_annual,
            "Vol Idiossincrática": risk.idiosyncratic_vol_annual,
            "R²": risk.r_squared,
        })

    if not rows:
        return None
    return pd.DataFrame(rows)
