"""Black-Litterman model.

Combines market-implied equilibrium returns (from CAPM / reverse-optimisation)
with subjective investor views to produce a posterior expected-return vector
and updated covariance matrix.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


# ------------------------------------------------------------------
# Data classes
# ------------------------------------------------------------------

@dataclass
class BLView:
    """A single investor view (absolute or relative)."""
    view_id: int
    view_type: str          # "absolute" | "relative"
    asset_1: str
    asset_2: str | None     # only for relative views
    expected_excess_return: float
    confidence: float       # 0-1
    enabled: bool = True


@dataclass(frozen=True)
class BLResult:
    prior_pi: pd.Series             # equilibrium returns
    posterior_mu: pd.Series          # adjusted returns
    posterior_cov: pd.DataFrame      # adjusted covariance
    optimal_weights: pd.Series      # BL optimal weights
    views_used: int


# ------------------------------------------------------------------
# Core BL Functions
# ------------------------------------------------------------------

def market_implied_prior_returns(
    cov: np.ndarray | pd.DataFrame,
    market_weights: np.ndarray | pd.Series,
    risk_aversion: float,
) -> np.ndarray:
    """Reverse-optimisation: π = δ Σ w_mkt."""
    S = np.asarray(cov, dtype=float)
    w = np.asarray(market_weights, dtype=float)
    return risk_aversion * S @ w


def build_pick_matrix(
    views: list[BLView],
    asset_universe: list[str],
) -> np.ndarray:
    """Construct the K×N pick matrix P."""
    n = len(asset_universe)
    k = len(views)
    P = np.zeros((k, n))
    idx_map = {a: i for i, a in enumerate(asset_universe)}

    for row_idx, v in enumerate(views):
        if v.asset_1 in idx_map:
            P[row_idx, idx_map[v.asset_1]] = 1.0
        if v.view_type == "relative" and v.asset_2 and v.asset_2 in idx_map:
            P[row_idx, idx_map[v.asset_2]] = -1.0

    return P


def build_view_vector(views: list[BLView]) -> np.ndarray:
    """Construct the K×1 view vector Q."""
    return np.array([v.expected_excess_return for v in views], dtype=float)


def build_omega(
    views: list[BLView],
    cov: np.ndarray,
    P: np.ndarray,
    tau: float,
    method: str = "proportional",
) -> np.ndarray:
    """Build Ω (view uncertainty matrix).

    Methods:
    - ``"proportional"``:  Ω_ii = 1/c_i × (P Σ Pᵀ)_ii  where c_i is confidence
    - ``"idzorek"``:       Simplified Idzorek (He & Litterman 1999 variant)
    """
    k = P.shape[0]
    tau_cov = tau * cov
    p_sigma_pt = P @ tau_cov @ P.T

    omega = np.zeros((k, k))
    for i, v in enumerate(views):
        conf = max(min(v.confidence, 0.999), 0.001)
        if method == "idzorek":
            # Idzorek: scale diagonal by (1 - confidence) / confidence
            omega[i, i] = p_sigma_pt[i, i] * (1 - conf) / conf
        else:
            # Proportional: higher confidence → smaller Ω entry
            omega[i, i] = p_sigma_pt[i, i] / conf

    return omega


def black_litterman_posterior(
    prior_pi: np.ndarray,
    cov: np.ndarray,
    P: np.ndarray,
    Q: np.ndarray,
    omega: np.ndarray,
    tau: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute BL posterior returns and covariance.

    Returns (posterior_mu, posterior_cov).
    """
    tau_cov = tau * cov
    tau_cov_inv = np.linalg.inv(tau_cov)
    omega_inv = np.linalg.inv(omega)

    # Posterior precision = (τΣ)⁻¹ + Pᵀ Ω⁻¹ P
    M = tau_cov_inv + P.T @ omega_inv @ P
    M_inv = np.linalg.inv(M)

    # Posterior mean
    posterior_mu = M_inv @ (tau_cov_inv @ prior_pi + P.T @ omega_inv @ Q)

    # Posterior covariance (for portfolio optimisation)
    posterior_cov = cov + M_inv

    return posterior_mu, posterior_cov


def optimize_bl_portfolio(
    expected_returns: np.ndarray,
    cov: np.ndarray,
    rf: float,
    n: int,
    bounds: tuple | None = None,
) -> np.ndarray:
    """Max-Sharpe optimisation using BL posterior returns."""
    from services.markowitz import solve_max_sharpe  # avoid circular
    result = solve_max_sharpe(expected_returns, cov, rf, n, bounds)
    return result.weights.values


# ------------------------------------------------------------------
# Views I/O (CSV)
# ------------------------------------------------------------------

def load_views_from_csv(path: str) -> list[BLView]:
    """Load views from a CSV."""
    try:
        df = pd.read_csv(path)
    except Exception:
        return []

    if df.empty:
        return []

    views: list[BLView] = []
    for _, row in df.iterrows():
        try:
            v = BLView(
                view_id=int(row.get("view_id", 0)),
                view_type=str(row.get("view_type", "absolute")),
                asset_1=str(row.get("asset_1", "")),
                asset_2=str(row.get("asset_2", "")) if pd.notna(row.get("asset_2")) else None,
                expected_excess_return=float(row.get("expected_excess_return", 0.0)),
                confidence=float(row.get("confidence", 0.5)),
                enabled=bool(row.get("enabled", True)),
            )
            views.append(v)
        except Exception:
            continue
    return views


def save_views_to_csv(views: list[BLView], path: str) -> None:
    """Persist views to CSV."""
    rows = []
    for v in views:
        rows.append({
            "view_id": v.view_id,
            "view_type": v.view_type,
            "asset_1": v.asset_1,
            "asset_2": v.asset_2 or "",
            "expected_excess_return": v.expected_excess_return,
            "confidence": v.confidence,
            "enabled": v.enabled,
        })
    pd.DataFrame(rows).to_csv(path, index=False)


# ------------------------------------------------------------------
# High-level runner
# ------------------------------------------------------------------

def run_black_litterman(
    cov: pd.DataFrame,
    market_weights: pd.Series,
    views: list[BLView],
    asset_universe: list[str],
    risk_aversion: float = 2.5,
    tau: float = 0.05,
    rf: float = 0.0,
    max_weight: float = 1.0,
    omega_method: str = "proportional",
) -> BLResult | None:
    """One-call entry point for the BL model."""
    enabled_views = [v for v in views if v.enabled]
    if not enabled_views:
        return None

    cov_arr = cov.reindex(index=asset_universe, columns=asset_universe).fillna(0.0).values
    w_mkt = market_weights.reindex(asset_universe).fillna(0.0).values
    n = len(asset_universe)

    # Ensure CovMatrix is PD
    eigvals = np.linalg.eigvals(cov_arr)
    if np.any(eigvals <= 0):
        cov_arr = cov_arr + np.eye(n) * 1e-8

    # Equilibrium returns
    pi = market_implied_prior_returns(cov_arr, w_mkt, risk_aversion)

    # BL matrices
    P = build_pick_matrix(enabled_views, asset_universe)
    Q = build_view_vector(enabled_views)
    omega = build_omega(enabled_views, cov_arr, P, tau, omega_method)

    # Posterior
    try:
        post_mu, post_cov = black_litterman_posterior(pi, cov_arr, P, Q, omega, tau)
    except np.linalg.LinAlgError:
        return None

    # Optimal weights
    bounds = tuple((0.0, max_weight) for _ in range(n))
    opt_w = optimize_bl_portfolio(post_mu, post_cov, rf, n, bounds)

    return BLResult(
        prior_pi=pd.Series(pi, index=asset_universe),
        posterior_mu=pd.Series(post_mu, index=asset_universe),
        posterior_cov=pd.DataFrame(post_cov, index=asset_universe, columns=asset_universe),
        optimal_weights=pd.Series(opt_w, index=asset_universe),
        views_used=len(enabled_views),
    )
