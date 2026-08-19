"""Order management: validation rules for buy/sell/subscribe/redeem."""
from __future__ import annotations

import pandas as pd


class OrderValidationError(Exception):
    """Raised when an order fails validation."""


def validate_buy(
    ticker: str,
    quantity: float,
    price: float,
    cash_balance: float,
    fees: float = 0.0,
    allow_margin: bool = False,
) -> None:
    """Raise ``OrderValidationError`` if the buy order is invalid."""
    if not ticker or not ticker.strip():
        raise OrderValidationError("Informe o ticker do ativo.")
    if quantity <= 0:
        raise OrderValidationError("A quantidade precisa ser maior que zero.")
    if price <= 0:
        raise OrderValidationError("O preço precisa ser maior que zero.")

    total_cost = quantity * price + fees
    if not allow_margin and total_cost > cash_balance:
        raise OrderValidationError(
            f"Caixa insuficiente. Custo da operação: R$ {total_cost:,.2f}, "
            f"caixa disponível: R$ {cash_balance:,.2f}."
        )


def validate_sell(
    ticker: str,
    quantity: float,
    price: float,
    current_positions: dict[str, dict],
) -> None:
    """Raise ``OrderValidationError`` if the sell order is invalid."""
    if not ticker or not ticker.strip():
        raise OrderValidationError("Informe o ticker do ativo.")
    if quantity <= 0:
        raise OrderValidationError("A quantidade precisa ser maior que zero.")
    if price <= 0:
        raise OrderValidationError("O preço precisa ser maior que zero.")

    pos = current_positions.get(ticker)
    if pos is None or pos["quantity"] <= 0:
        raise OrderValidationError(f"Você não possui posição em {ticker}.")
    if quantity > pos["quantity"]:
        raise OrderValidationError(
            f"Quantidade disponível em {ticker}: {pos['quantity']:.2f}. "
            f"Você tentou vender {quantity:.2f}."
        )


def validate_subscription(amount: float) -> None:
    if amount <= 0:
        raise OrderValidationError("O valor do aporte precisa ser maior que zero.")


def validate_redemption(amount: float, pl_total: float) -> None:
    if amount <= 0:
        raise OrderValidationError("O valor do resgate precisa ser maior que zero.")
    if amount > pl_total * 1.001:  # small tolerance
        raise OrderValidationError(
            f"Resgate de R$ {amount:,.2f} excede o PL atual de R$ {pl_total:,.2f}."
        )
