"""Operações page — Home Broker + NAV Engine UI.

Allows registering buy/sell/subscribe/redeem operations and displays
positions, cash ledger, NAV history and charts.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.transactions_repository import TransactionsRepository
from services.nav_engine import (
    compute_cash_balance,
    compute_nav_record,
    compute_portfolio_pl,
    compute_positions_from_trades,
    mark_to_market_positions,
    process_redemption,
    process_subscription,
)
from services.order_management import (
    OrderValidationError,
    validate_buy,
    validate_redemption,
    validate_sell,
    validate_subscription,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _get_repo() -> TransactionsRepository:
    return TransactionsRepository(base_path=_PROJECT_ROOT)


def _latest_prices_from_session() -> dict[str, float]:
    """Try to pull latest prices from the historical data in session state."""
    hist = st.session_state.get("historical_df")
    if hist is None or hist.empty:
        return {}
    last_row = hist.ffill().iloc[-1]
    return {col: float(val) for col, val in last_row.items() if pd.notna(val)}


def _format_brl(value: float) -> str:
    return f"R$ {value:,.2f}"


# ------------------------------------------------------------------
# Render
# ------------------------------------------------------------------

def render_operations_page() -> None:
    repo = _get_repo()

    st.title("Operações")
    st.caption("Registre compras, vendas, aportes e resgates. Acompanhe posições, caixa e evolução de cotas.")

    trades_df = repo.load_transactions()
    cash_df = repo.load_cash_ledger()
    nav_df = repo.load_nav_history()

    positions = compute_positions_from_trades(trades_df)
    cash_balance = compute_cash_balance(cash_df)
    latest_prices = _latest_prices_from_session()
    positions_mtm = mark_to_market_positions(positions, latest_prices)
    pl_total = compute_portfolio_pl(positions_mtm, cash_balance)

    # Units
    units_outstanding = 1000.0  # default
    if not nav_df.empty and "units_outstanding" in nav_df.columns:
        last_units = pd.to_numeric(nav_df["units_outstanding"], errors="coerce").dropna()
        if not last_units.empty:
            units_outstanding = float(last_units.iloc[-1])

    nav_per_unit = pl_total / units_outstanding if units_outstanding > 0 else 1.0

    # ── KPI row ──────────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Patrimônio Líquido", _format_brl(pl_total))
    k2.metric("Caixa", _format_brl(cash_balance))
    k3.metric("Cota", f"{nav_per_unit:.6f}")
    k4.metric("Cotas emitidas", f"{units_outstanding:,.2f}")

    st.markdown("---")

    # ── Operation forms ──────────────────────────────────────────
    tab_buy, tab_sell, tab_sub, tab_red = st.tabs(["🛒 Compra", "💰 Venda", "📥 Aporte", "📤 Resgate"])

    with tab_buy:
        with st.form("form_buy", clear_on_submit=True):
            bc1, bc2, bc3 = st.columns(3)
            buy_ticker = bc1.text_input("Ticker", placeholder="PETR4").strip().upper()
            buy_qty = bc2.number_input("Quantidade", min_value=1, value=100)
            buy_price = bc3.number_input("Preço (R$)", min_value=0.01, value=30.0, format="%.2f")
            buy_fees = st.number_input("Custos (R$)", min_value=0.0, value=0.0, format="%.2f", key="buy_fees")
            buy_notes = st.text_input("Notas", key="buy_notes")
            submitted = st.form_submit_button("Registrar compra", type="primary")
            if submitted:
                try:
                    from services.portfolio_analytics import normalize_equity_ticker
                    norm_ticker = normalize_equity_ticker(buy_ticker)
                    validate_buy(norm_ticker, buy_qty, buy_price, cash_balance, buy_fees)
                    repo.add_transaction(norm_ticker, "BUY", buy_qty, buy_price, fees=buy_fees, notes=buy_notes)
                    repo.add_cash_event("compra", -(buy_qty * buy_price + buy_fees), reference_id=norm_ticker, notes=buy_notes)
                    st.success(f"Compra de {buy_qty} {norm_ticker} a R$ {buy_price:.2f} registrada.")
                    st.rerun()
                except (OrderValidationError, Exception) as e:
                    st.error(str(e))

    with tab_sell:
        with st.form("form_sell", clear_on_submit=True):
            sc1, sc2, sc3 = st.columns(3)
            sell_ticker = sc1.text_input("Ticker", placeholder="PETR4", key="sell_t").strip().upper()
            sell_qty = sc2.number_input("Quantidade", min_value=1, value=100, key="sell_q")
            sell_price = sc3.number_input("Preço (R$)", min_value=0.01, value=30.0, format="%.2f", key="sell_p")
            sell_fees = st.number_input("Custos (R$)", min_value=0.0, value=0.0, format="%.2f", key="sell_fees")
            sell_notes = st.text_input("Notas", key="sell_notes")
            submitted = st.form_submit_button("Registrar venda", type="primary")
            if submitted:
                try:
                    from services.portfolio_analytics import normalize_equity_ticker
                    norm_ticker = normalize_equity_ticker(sell_ticker)
                    validate_sell(norm_ticker, sell_qty, sell_price, positions)
                    repo.add_transaction(norm_ticker, "SELL", sell_qty, sell_price, fees=sell_fees, notes=sell_notes)
                    repo.add_cash_event("venda", sell_qty * sell_price - sell_fees, reference_id=norm_ticker, notes=sell_notes)
                    st.success(f"Venda de {sell_qty} {norm_ticker} a R$ {sell_price:.2f} registrada.")
                    st.rerun()
                except (OrderValidationError, Exception) as e:
                    st.error(str(e))

    with tab_sub:
        with st.form("form_subscription", clear_on_submit=True):
            sub_amount = st.number_input("Valor do aporte (R$)", min_value=0.01, value=10000.0, format="%.2f")
            sub_notes = st.text_input("Notas", key="sub_notes")
            submitted = st.form_submit_button("Registrar aporte", type="primary")
            if submitted:
                try:
                    validate_subscription(sub_amount)
                    new_units, _ = process_subscription(sub_amount, nav_per_unit)
                    repo.add_cash_event("aporte", sub_amount, notes=sub_notes)
                    # Update units
                    new_total_units = units_outstanding + new_units
                    new_pl = pl_total + sub_amount
                    record = compute_nav_record(new_pl, cash_balance + sub_amount, new_total_units, net_flow=sub_amount, prev_nav=nav_per_unit)
                    repo.append_nav_record(record)
                    st.success(f"Aporte de R$ {sub_amount:,.2f} registrado. {new_units:,.4f} cotas emitidas.")
                    st.rerun()
                except (OrderValidationError, Exception) as e:
                    st.error(str(e))

    with tab_red:
        with st.form("form_redemption", clear_on_submit=True):
            red_amount = st.number_input("Valor do resgate (R$)", min_value=0.01, value=5000.0, format="%.2f")
            red_notes = st.text_input("Notas", key="red_notes")
            submitted = st.form_submit_button("Registrar resgate", type="primary")
            if submitted:
                try:
                    validate_redemption(red_amount, pl_total)
                    units_cancelled, _ = process_redemption(red_amount, nav_per_unit)
                    repo.add_cash_event("resgate", -red_amount, notes=red_notes)
                    new_total_units = max(units_outstanding - units_cancelled, 0)
                    new_pl = pl_total - red_amount
                    record = compute_nav_record(new_pl, cash_balance - red_amount, new_total_units, net_flow=-red_amount, prev_nav=nav_per_unit)
                    repo.append_nav_record(record)
                    st.success(f"Resgate de R$ {red_amount:,.2f} registrado. {units_cancelled:,.4f} cotas canceladas.")
                    st.rerun()
                except (OrderValidationError, Exception) as e:
                    st.error(str(e))

    st.markdown("---")

    # ── Positions table ──────────────────────────────────────────
    st.subheader("Posição consolidada")
    if positions_mtm:
        pos_rows = [
            {
                "Ticker": p.ticker.replace(".SA", ""),
                "Qtd": p.quantity,
                "Custo Médio": p.avg_cost,
                "Preço Mercado": p.market_price,
                "Valor": p.market_value,
                "P&L": p.pnl,
                "P&L (%)": p.pnl_pct,
            }
            for p in positions_mtm
        ]
        pos_df = pd.DataFrame(pos_rows).sort_values("Valor", ascending=False)
        st.dataframe(
            pos_df.style.format({
                "Qtd": "{:,.0f}",
                "Custo Médio": "R$ {:,.2f}",
                "Preço Mercado": "R$ {:,.2f}",
                "Valor": "R$ {:,.2f}",
                "P&L": "R$ {:,.2f}",
                "P&L (%)": "{:.2%}",
            }),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("Nenhuma posição registrada. Utilize os formulários acima para registrar operações.")

    # ── Transactions history ─────────────────────────────────────
    st.subheader("Histórico de operações")
    if not trades_df.empty:
        display_cols = ["timestamp", "ticker", "side", "quantity", "price", "gross_value", "fees", "net_value", "notes"]
        show_cols = [c for c in display_cols if c in trades_df.columns]
        st.dataframe(trades_df[show_cols].sort_values("timestamp", ascending=False), use_container_width=True, hide_index=True)
    else:
        st.info("Nenhuma operação registrada.")

    # ── Cash ledger ──────────────────────────────────────────────
    st.subheader("Movimentação de caixa")
    if not cash_df.empty:
        st.dataframe(cash_df.sort_values("timestamp", ascending=False), use_container_width=True, hide_index=True)
    else:
        st.info("Nenhuma movimentação de caixa registrada.")

    # ── NAV Chart ────────────────────────────────────────────────
    st.subheader("Evolução da Cota")
    if not nav_df.empty and len(nav_df) >= 2:
        nav_df["date"] = pd.to_datetime(nav_df["date"], errors="coerce")
        nav_df["nav_per_unit"] = pd.to_numeric(nav_df["nav_per_unit"], errors="coerce")
        valid = nav_df.dropna(subset=["date", "nav_per_unit"])
        if len(valid) >= 2:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=valid["date"], y=valid["nav_per_unit"],
                mode="lines+markers",
                line=dict(color="#4979f6", width=2),
                marker=dict(size=5),
                name="Cota",
            ))
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#6B7280", size=11, family="Inter"),
                margin=dict(l=0, r=20, t=30, b=30),
                yaxis_title="Valor da cota",
            )
            fig.update_xaxes(showgrid=True, gridcolor="#E5E7EB")
            fig.update_yaxes(showgrid=True, gridcolor="#E5E7EB")
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        else:
            st.info("Registre pelo menos duas movimentações para visualizar a evolução da cota.")
    else:
        st.info("Registre pelo menos duas movimentações para visualizar a evolução da cota.")
