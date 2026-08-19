"""Markowitz mean-variance optimisation service.

Supports three return-estimation modes (historical, CAPM forward-looking,
Black-Litterman posterior) and two portfolio-construction modes (Monte-Carlo
simulation, deterministic SLSQP optimisation with efficient frontier).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize


TRADING_DAYS_PER_YEAR = 252


# ------------------------------------------------------------------
# Data classes
# ------------------------------------------------------------------

@dataclass(frozen=True)
class OptimizedPortfolio:
    weights: pd.Series
    expected_return: float
    volatility: float
    sharpe: float


@dataclass(frozen=True)
class MarkowitzFullResult:
    """Aggregated result for the Markowitz page."""
    # Monte-Carlo cloud (may be empty in optimisation-only mode)
    cloud: pd.DataFrame               # columns: retorno, volatilidade, sharpe, + asset weights
    # Deterministic efficient frontier
    frontier: pd.DataFrame             # columns: retorno, volatilidade
    # Key portfolios
    min_variance: OptimizedPortfolio
    max_sharpe: OptimizedPortfolio
    tangency: OptimizedPortfolio | None
    # Metadata
    mu: pd.Series                      # expected return vector used
    cov: pd.DataFrame                  # covariance matrix used
    rf: float
    asset_names: list[str]


# ------------------------------------------------------------------
# Return estimation
# ------------------------------------------------------------------

def estimate_expected_returns(
    prices: pd.DataFrame,
    method: str = "historical",
    *,
    rf_annual: float = 0.0,
    emrp_annual: float = 0.06,
    bl_posterior: pd.Series | None = None,
) -> pd.Series:
    """Compute annualised expected return vector *mu*.

    ``method`` can be ``"historical"`` | ``"capm"`` | ``"bl_posterior"``.
    """
    returns = prices.pct_change().dropna()
    if method == "bl_posterior" and bl_posterior is not None:
        return bl_posterior.reindex(prices.columns).fillna(0.0)

    if method == "capm":
        from services.capm import estimate_beta, capm_expected_return  # local to avoid circular
        if "IBOVESPA" in returns.columns:
            market_ret = returns["IBOVESPA"]
        else:
            market_ret = returns.mean(axis=1)  # fallback: equal-weight proxy

        mu_dict: dict[str, float] = {}
        for col in prices.columns:
            if col == "IBOVESPA":
                continue
            beta = estimate_beta(returns[col], market_ret)
            mu_dict[col] = capm_expected_return(rf_annual, beta, emrp_annual)
        return pd.Series(mu_dict)

    # Default: historical annualised mean
    return returns.mean() * TRADING_DAYS_PER_YEAR


def estimate_covariance(prices: pd.DataFrame) -> pd.DataFrame:
    """Annualised sample covariance (with tiny ridge for PD guarantee)."""
    returns = prices.pct_change().dropna()
    cov = returns.cov() * TRADING_DAYS_PER_YEAR
    n = cov.shape[0]
    eigenvalues = np.linalg.eigvals(cov.values)
    if np.any(eigenvalues <= 0):
        cov = cov + pd.DataFrame(np.eye(n) * 1e-8, index=cov.index, columns=cov.columns)
    return cov


# ------------------------------------------------------------------
# Single-portfolio helpers
# ------------------------------------------------------------------

def portfolio_return(weights: np.ndarray, mu: np.ndarray) -> float:
    return float(weights @ mu)


def portfolio_volatility(weights: np.ndarray, cov: np.ndarray) -> float:
    return float(np.sqrt(weights @ cov @ weights))


# ------------------------------------------------------------------
# Optimisation
# ------------------------------------------------------------------

def _default_constraints(n: int, bounds: tuple | None = None):
    if bounds is None:
        bounds = tuple((0.0, 1.0) for _ in range(n))
    eq_sum = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    return bounds, [eq_sum]


def solve_min_variance(
    cov: np.ndarray,
    n: int,
    bounds: tuple | None = None,
) -> OptimizedPortfolio:
    bounds, constraints = _default_constraints(n, bounds)
    w0 = np.ones(n) / n

    def obj(w):
        return portfolio_volatility(w, cov)

    res = minimize(obj, w0, method="SLSQP", bounds=bounds, constraints=constraints,
                   options={"ftol": 1e-12, "maxiter": 2000})
    w = res.x
    vol = portfolio_volatility(w, cov)
    return OptimizedPortfolio(weights=pd.Series(w), expected_return=0.0, volatility=vol, sharpe=0.0)


def solve_max_sharpe(
    mu: np.ndarray,
    cov: np.ndarray,
    rf: float,
    n: int,
    bounds: tuple | None = None,
) -> OptimizedPortfolio:
    bounds, constraints = _default_constraints(n, bounds)
    w0 = np.ones(n) / n

    def neg_sharpe(w):
        ret = portfolio_return(w, mu)
        vol = portfolio_volatility(w, cov)
        return -(ret - rf) / vol if vol > 1e-12 else 0.0

    res = minimize(neg_sharpe, w0, method="SLSQP", bounds=bounds, constraints=constraints,
                   options={"ftol": 1e-12, "maxiter": 2000})
    w = res.x
    ret = portfolio_return(w, mu)
    vol = portfolio_volatility(w, cov)
    sharpe = (ret - rf) / vol if vol > 1e-12 else 0.0
    return OptimizedPortfolio(weights=pd.Series(w), expected_return=ret, volatility=vol, sharpe=sharpe)


def solve_target_return(
    mu: np.ndarray,
    cov: np.ndarray,
    target: float,
    n: int,
    bounds: tuple | None = None,
) -> np.ndarray | None:
    bounds, constraints = _default_constraints(n, bounds)
    constraints.append({"type": "eq", "fun": lambda w, t=target: portfolio_return(w, mu) - t})
    w0 = np.ones(n) / n

    res = minimize(
        lambda w: portfolio_volatility(w, cov),
        w0, method="SLSQP", bounds=bounds, constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 2000},
    )
    return res.x if res.success else None


def efficient_frontier(
    mu: np.ndarray,
    cov: np.ndarray,
    rf: float,
    n: int,
    n_points: int = 50,
    bounds: tuple | None = None,
) -> pd.DataFrame:
    """Deterministic efficient frontier via target-return sweep."""
    # Get min-var portfolio to find range
    mv = solve_min_variance(cov, n, bounds)
    mv_ret = portfolio_return(mv.weights.values, mu)

    target_max = float(mu.max()) * 1.05
    target_min = float(mu.min()) * 0.95
    if target_min > mv_ret:
        target_min = mv_ret * 0.95

    targets = np.linspace(target_min, target_max, n_points)

    # Split: upper (from min-vol upward), lower (from min-vol downward)
    mid_idx = int(np.argmin(np.abs(targets - mv_ret)))

    results: list[dict] = []

    # Upper half (warm-started from min-vol)
    prev_w = mv.weights.values.copy()
    for t in targets[mid_idx:]:
        w = _solve_with_warmstart(mu, cov, t, prev_w, n, bounds)
        if w is not None:
            results.append({"retorno": portfolio_return(w, mu), "volatilidade": portfolio_volatility(w, cov)})
            prev_w = w.copy()

    # Lower half (warm-started from min-vol, going down)
    lower: list[dict] = []
    prev_w = mv.weights.values.copy()
    for t in reversed(targets[:mid_idx]):
        w = _solve_with_warmstart(mu, cov, t, prev_w, n, bounds)
        if w is not None:
            lower.append({"retorno": portfolio_return(w, mu), "volatilidade": portfolio_volatility(w, cov)})
            prev_w = w.copy()

    lower.reverse()
    return pd.DataFrame(lower + results)


def _solve_with_warmstart(mu, cov, target, w0, n, bounds):
    if bounds is None:
        bounds = tuple((0.0, 1.0) for _ in range(n))
    constraints = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
        {"type": "eq", "fun": lambda w, t=target: portfolio_return(w, mu) - t},
    ]
    res = minimize(
        lambda w: portfolio_volatility(w, cov),
        w0, method="SLSQP", bounds=bounds, constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 2000},
    )
    return res.x if res.success else None


def tangency_portfolio(
    mu: np.ndarray,
    cov: np.ndarray,
    rf: float,
    n: int,
    bounds: tuple | None = None,
) -> OptimizedPortfolio:
    """Tangency portfolio = max Sharpe portfolio."""
    return solve_max_sharpe(mu, cov, rf, n, bounds)


# ------------------------------------------------------------------
# Monte-Carlo simulation
# ------------------------------------------------------------------

def simulate_portfolios(
    mu: np.ndarray,
    cov: np.ndarray,
    asset_names: list[str],
    n_sims: int = 10000,
    alpha: float = 0.5,
    seed: int = 42,
) -> pd.DataFrame:
    """Dirichlet Monte-Carlo cloud."""
    n = len(asset_names)
    rng = np.random.default_rng(seed)
    weights = rng.dirichlet(np.ones(n) * alpha, size=n_sims)
    rets = weights @ mu
    vols = np.array([portfolio_volatility(w, cov) for w in weights])
    sharpes = np.where(vols > 1e-12, rets / vols, 0.0)

    data = np.column_stack([rets, vols, sharpes, weights])
    return pd.DataFrame(data, columns=["retorno", "volatilidade", "sharpe", *asset_names])


# ------------------------------------------------------------------
# High-level runner
# ------------------------------------------------------------------

def run_markowitz(
    prices: pd.DataFrame,
    asset_names: list[str],
    rf: float = 0.0,
    return_method: str = "historical",
    emrp: float = 0.06,
    bl_posterior: pd.Series | None = None,
    max_weight: float = 1.0,
    run_simulation: bool = True,
    n_frontier_points: int = 50,
    n_sims: int = 10000,
) -> MarkowitzFullResult | None:
    """One-call entry point used by the UI."""
    if len(asset_names) < 2:
        return None

    asset_prices = prices[asset_names].ffill().bfill()
    returns = asset_prices.pct_change().dropna()
    if len(returns) < 21:
        return None

    mu_series = estimate_expected_returns(
        asset_prices, method=return_method,
        rf_annual=rf, emrp_annual=emrp, bl_posterior=bl_posterior,
    )
    mu_series = mu_series.reindex(asset_names).fillna(0.0)

    cov_df = estimate_covariance(asset_prices)
    cov_df = cov_df.reindex(index=asset_names, columns=asset_names).fillna(0.0)

    mu_arr = mu_series.values
    cov_arr = cov_df.values
    n = len(asset_names)
    bounds = tuple((0.0, max_weight) for _ in range(n))

    # Key portfolios
    mv = solve_min_variance(cov_arr, n, bounds)
    mv = OptimizedPortfolio(
        weights=pd.Series(mv.weights.values, index=asset_names),
        expected_return=portfolio_return(mv.weights.values, mu_arr),
        volatility=mv.volatility,
        sharpe=(portfolio_return(mv.weights.values, mu_arr) - rf) / mv.volatility if mv.volatility > 1e-12 else 0.0,
    )

    ms = solve_max_sharpe(mu_arr, cov_arr, rf, n, bounds)
    ms = OptimizedPortfolio(
        weights=pd.Series(ms.weights.values, index=asset_names),
        expected_return=ms.expected_return,
        volatility=ms.volatility,
        sharpe=ms.sharpe,
    )

    tg = tangency_portfolio(mu_arr, cov_arr, rf, n, bounds)
    tg = OptimizedPortfolio(
        weights=pd.Series(tg.weights.values, index=asset_names),
        expected_return=tg.expected_return,
        volatility=tg.volatility,
        sharpe=tg.sharpe,
    )

    # Frontier
    frontier_df = efficient_frontier(mu_arr, cov_arr, rf, n, n_frontier_points, bounds)

    # Simulation cloud
    cloud_df = pd.DataFrame()
    if run_simulation:
        cloud_df = simulate_portfolios(mu_arr, cov_arr, asset_names, n_sims)

    return MarkowitzFullResult(
        cloud=cloud_df,
        frontier=frontier_df,
        min_variance=mv,
        max_sharpe=ms,
        tangency=tg,
        mu=mu_series,
        cov=cov_df,
        rf=rf,
        asset_names=asset_names,
    )
