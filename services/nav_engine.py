"""NAV / quota engine.

Computes positions from trades, marks-to-market, handles subscriptions
and redemptions, and maintains a fund NAV history.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd


# ------------------------------------------------------------------
# Data classes
# ------------------------------------------------------------------

@dataclass
class Position:
    ticker: str
    quantity: float
    avg_cost: float
    market_price: float
    market_value: float
    pnl: float
    pnl_pct: float


# ------------------------------------------------------------------
# Core functions
# ------------------------------------------------------------------

def compute_cash_balance(cash_ledger: pd.DataFrame) -> float:
    """Sum of all cash events."""
    if cash_ledger.empty or "amount" not in cash_ledger.columns:
        return 0.0
    return float(pd.to_numeric(cash_ledger["amount"], errors="coerce").fillna(0).sum())


def compute_positions_from_trades(trades: pd.DataFrame) -> dict[str, dict]:
    """Aggregate trades into positions with average cost.

    Returns {ticker: {"quantity": float, "avg_cost": float}}.
    """
    if trades.empty:
        return {}

    positions: dict[str, dict] = {}
    for _, row in trades.iterrows():
        ticker = str(row.get("ticker", ""))
        side = str(row.get("side", "")).upper()
        qty = float(pd.to_numeric(row.get("quantity"), errors="coerce") or 0)
        price = float(pd.to_numeric(row.get("price"), errors="coerce") or 0)
        if not ticker or qty <= 0:
            continue

        if ticker not in positions:
            positions[ticker] = {"quantity": 0.0, "avg_cost": 0.0}

        pos = positions[ticker]
        if side in ("BUY", "SUBSCRIPTION"):
            total_cost = pos["avg_cost"] * pos["quantity"] + price * qty
            pos["quantity"] += qty
            pos["avg_cost"] = total_cost / pos["quantity"] if pos["quantity"] > 0 else 0.0
        elif side in ("SELL", "REDEMPTION"):
            pos["quantity"] = max(pos["quantity"] - qty, 0.0)
            # avg_cost stays the same on sells

    # Remove zeroed-out positions
    return {t: p for t, p in positions.items() if p["quantity"] > 0}


def mark_to_market_positions(
    positions: dict[str, dict],
    latest_prices: dict[str, float],
) -> list[Position]:
    """Mark each position to market and compute PnL."""
    result: list[Position] = []
    for ticker, pos in positions.items():
        mkt_price = latest_prices.get(ticker, pos["avg_cost"])
        mkt_value = pos["quantity"] * mkt_price
        cost_basis = pos["quantity"] * pos["avg_cost"]
        pnl = mkt_value - cost_basis
        pnl_pct = pnl / cost_basis if cost_basis > 0 else 0.0
        result.append(Position(
            ticker=ticker,
            quantity=pos["quantity"],
            avg_cost=pos["avg_cost"],
            market_price=mkt_price,
            market_value=mkt_value,
            pnl=pnl,
            pnl_pct=pnl_pct,
        ))
    return result


def compute_portfolio_pl(positions_mtm: list[Position], cash_balance: float) -> float:
    """Total PL = cash + sum of market values."""
    return cash_balance + sum(p.market_value for p in positions_mtm)


def process_subscription(amount: float, nav_per_unit_prev: float) -> tuple[float, float]:
    """Process a cash subscription.

    Returns (new_units_issued, updated_nav).
    """
    if nav_per_unit_prev <= 0:
        nav_per_unit_prev = 1.0
    new_units = amount / nav_per_unit_prev
    return new_units, nav_per_unit_prev  # NAV doesn't change on subscription


def process_redemption(amount: float, nav_per_unit_prev: float) -> tuple[float, float]:
    """Process a cash redemption.

    Returns (units_cancelled, updated_nav).
    """
    if nav_per_unit_prev <= 0:
        nav_per_unit_prev = 1.0
    units_cancelled = amount / nav_per_unit_prev
    return units_cancelled, nav_per_unit_prev


def compute_nav_record(
    pl_total: float,
    cash_balance: float,
    units_outstanding: float,
    net_flow: float = 0.0,
    prev_nav: float | None = None,
) -> dict:
    """Build a single NAV history row."""
    nav = pl_total / units_outstanding if units_outstanding > 0 else 1.0
    gross_return = (nav / prev_nav - 1) if prev_nav and prev_nav > 0 else 0.0
    return {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "pl_total": pl_total,
        "cash_balance": cash_balance,
        "units_outstanding": units_outstanding,
        "nav_per_unit": nav,
        "net_flow": net_flow,
        "gross_return": gross_return,
        "net_return": gross_return,  # simplified (no management fee for now)
    }
