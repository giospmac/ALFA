"""Black-Litterman page.

Dedicated page for the Black-Litterman allocation model:
- Equilibrium returns (reverse-optimisation)
- Editable views table with confidence
- Posterior returns and optimal weights
- Comparison chart and table
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.config_repository import ConfigRepository
from core.portfolio_repository import PortfolioRepository
from services.black_litterman import BLView, run_black_litterman
from services.markowitz import (
    estimate_covariance,
    estimate_expected_returns,
    run_markowitz,
)
from services.portfolio_analytics import (
    _available_asset_columns,
    _normalized_weights,
)


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_CUSTOM_BLUES = [[0.0, "#93c5fd"], [0.35, "#4979f6"], [0.7, "#1e379b"], [1.0, "#102170"]]


def _load_data():
    if "portfolio_df" in st.session_state and "historical_df" in st.session_state:
        return st.session_state["portfolio_df"], st.session_state["historical_df"]
    repo = PortfolioRepository(base_path=_PROJECT_ROOT)
    snapshot = repo.load_snapshot()
    return snapshot.portfolio, snapshot.historical


def render_black_litterman_page() -> None:
    portfolio_df, historical_df = _load_data()

    st.title("Black-Litterman")
    st.caption("Combine retornos de equilíbrio com suas visões para gerar alocações ótimas.")

    if portfolio_df.empty or historical_df.empty:
        st.info("Monte o portfolio e atualize o histórico na página principal.")
        return

    asset_names = _available_asset_columns(portfolio_df, historical_df)
    if len(asset_names) < 2:
        st.info("São necessários pelo menos dois ativos válidos.")
        return

    # ── Parameters ───────────────────────────────────────────────
    config_repo = ConfigRepository(base_path=_PROJECT_ROOT)
    config = config_repo.load()

    with st.expander("Configurações Básicas", expanded=False):
        st.write("Defina as regras básicas de como o modelo deve balancear o seu dinheiro.")
        p1, p2 = st.columns(2)
        rf_val = p1.number_input(
            "Rendimento Seguro (Selic a.a.)", 
            value=config.risk_free_rate, min_value=0.0, max_value=1.0, step=0.0025, format="%.4f", key="bl_rf",
            help="Taxa de rendimento livre de risco (ex: Tesouro Selic atual)."
        )
        max_w = p2.number_input(
            "Concentração Máxima por Ativo", 
            value=config.max_weight, min_value=0.05, max_value=1.0, step=0.05, format="%.2f", key="bl_maxw",
            help="Limite de quanto do seu dinheiro pode ficar em uma única ação (ex: 1.00 = 100%)."
        )
        
        use_current = st.checkbox(
            "Usar minha carteira atual como ponto de partida neutro", 
            value=True,
            help="O modelo vai tentar respeitar os pesos que você já tem hoje, alterando apenas os ativos sobre os quais você colocar uma 'Visão' abaixo."
        )

        with st.expander("🔧 Modo Avançado (Nerd)"):
            st.caption("Ajustes finos do algoritmo. Se não souber o que são, deixe como estão!")
            a1, a2 = st.columns(2)
            delta_val = a1.number_input("Aversão ao Risco (δ)", value=config.delta, min_value=0.1, max_value=10.0, step=0.1, format="%.2f", key="bl_delta")
            tau_val = a2.number_input("Peso do Histórico (τ)", value=config.tau, min_value=0.001, max_value=0.5, step=0.005, format="%.3f", key="bl_tau")

        if delta_val != config.delta or tau_val != config.tau or rf_val != config.risk_free_rate or max_w != config.max_weight:
            config_repo.update(risk_free_rate=rf_val, delta=delta_val, tau=tau_val, max_weight=max_w)

    # ── Views editor ─────────────────────────────────────────────
    st.subheader("Suas Visões de Mercado")
    st.caption('Como você acha que seus investimentos vão render? Adicione sua expectativa aqui (ex: "Acho que BBAS3 vai render 14% ao ano" ou "Acho que VALE vai render mais que PETR").')

    views_key = "bl_views_editor_page"
    if views_key not in st.session_state:
        st.session_state[views_key] = pd.DataFrame({
            "Tipo": ["absolute"],
            "Ativo 1": [asset_names[0].replace(".SA", "") if asset_names else ""],
            "Ativo 2": [""],
            "Retorno esperado (a.a.)": [0.14],
            "Confiança (0-1)": [0.7],
            "Ativo": [True],
        })

    edited_views = st.data_editor(
        st.session_state[views_key],
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "Tipo": st.column_config.SelectboxColumn(options=["absolute", "relative"]),
            "Retorno esperado (a.a.)": st.column_config.NumberColumn(format="%.4f"),
            "Confiança (0-1)": st.column_config.NumberColumn(min_value=0.01, max_value=0.99, format="%.2f"),
        },
    )
    st.session_state[views_key] = edited_views

    # Parse views
    bl_views: list[BLView] = []
    for idx, row in edited_views.iterrows():
        if not row.get("Ativo", True):
            continue
        a1 = str(row.get("Ativo 1", "")).strip()
        if not a1:
            continue
        a1_full = a1 if a1 in asset_names else f"{a1}.SA" if f"{a1}.SA" in asset_names else a1
        a2_raw = str(row.get("Ativo 2", "")).strip()
        a2_full = None
        if a2_raw:
            a2_full = a2_raw if a2_raw in asset_names else f"{a2_raw}.SA" if f"{a2_raw}.SA" in asset_names else a2_raw

        bl_views.append(BLView(
            view_id=idx,
            view_type=str(row.get("Tipo", "absolute")),
            asset_1=a1_full,
            asset_2=a2_full,
            expected_excess_return=float(row.get("Retorno esperado (a.a.)", 0.0)),
            confidence=float(row.get("Confiança (0-1)", 0.5)),
            enabled=True,
        ))

    if not bl_views:
        st.info("Adicione pelo menos uma view para executar o modelo.")
        return

    # ── Run BL ───────────────────────────────────────────────────
    prices = historical_df[asset_names].ffill().bfill()
    cov_df = estimate_covariance(prices)
    mkt_w = (
        _normalized_weights(portfolio_df, asset_names)
        if use_current
        else pd.Series(np.ones(len(asset_names)) / len(asset_names), index=asset_names)
    )

    bl_result = run_black_litterman(
        cov=cov_df,
        market_weights=mkt_w,
        views=bl_views,
        asset_universe=asset_names,
        risk_aversion=delta_val,
        tau=tau_val,
        rf=rf_val,
        max_weight=max_w,
    )

    if bl_result is None:
        st.warning("Não foi possível calcular o modelo. Verifique se as views referenciam ativos válidos.")
        return

    st.markdown("---")

    # ── KPIs ─────────────────────────────────────────────────────
    bl_weights = bl_result.optimal_weights
    bl_ret = float(bl_weights.values @ bl_result.posterior_mu.reindex(asset_names).fillna(0).values)
    bl_vol = float(np.sqrt(bl_weights.values @ cov_df.reindex(index=asset_names, columns=asset_names).fillna(0).values @ bl_weights.values))
    bl_sharpe = (bl_ret - rf_val) / bl_vol if bl_vol > 1e-12 else 0.0

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Retorno esperado (BL)", f"{bl_ret:.2%}")
    k2.metric("Volatilidade", f"{bl_vol:.2%}")
    k3.metric("Sharpe", f"{bl_sharpe:.2f}")
    k4.metric("Views utilizadas", str(bl_result.views_used))

    # ── Comparison: Equilibrium vs Posterior ──────────────────────
    st.subheader("Retornos: Equilíbrio vs. Posterior")
    comp_df = pd.DataFrame({
        "Ativo": [a.replace(".SA", "") for a in asset_names],
        "Equilíbrio (π)": bl_result.prior_pi.values,
        "Posterior (BL)": bl_result.posterior_mu.values,
        "Diferença": bl_result.posterior_mu.values - bl_result.prior_pi.values,
    })

    # Bar chart
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=comp_df["Ativo"], y=comp_df["Equilíbrio (π)"],
        name="Equilíbrio", marker_color="#93c5fd",
    ))
    fig.add_trace(go.Bar(
        x=comp_df["Ativo"], y=comp_df["Posterior (BL)"],
        name="Posterior (BL)", marker_color="#4979f6",
    ))
    fig.update_layout(
        barmode="group",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#6B7280", size=11, family="Inter"),
        margin=dict(l=0, r=20, t=30, b=30),
        yaxis=dict(title="Retorno anualizado", tickformat=".1%", showgrid=True, gridcolor="#E5E7EB"),
        xaxis=dict(showgrid=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # Table
    st.dataframe(
        comp_df.style.format({
            "Equilíbrio (π)": "{:.2%}",
            "Posterior (BL)": "{:.2%}",
            "Diferença": "{:+.2%}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    # ── Optimal weights ──────────────────────────────────────────
    st.subheader("Pesos ótimos Black-Litterman")
    weights_df = pd.DataFrame({
        "Ativo": [a.replace(".SA", "") for a in asset_names],
        "Peso atual (%)": mkt_w.reindex(asset_names).fillna(0).values * 100,
        "Peso BL ótimo (%)": bl_weights.reindex(asset_names).fillna(0).values * 100,
        "Δ Peso (p.p.)": (bl_weights.reindex(asset_names).fillna(0).values - mkt_w.reindex(asset_names).fillna(0).values) * 100,
    })
    weights_df = weights_df[
        (weights_df["Peso atual (%)"].abs() > 0.1) | (weights_df["Peso BL ótimo (%)"].abs() > 0.1)
    ].sort_values("Peso BL ótimo (%)", ascending=False)

    st.dataframe(
        weights_df.style.format({
            "Peso atual (%)": "{:.2f}",
            "Peso BL ótimo (%)": "{:.2f}",
            "Δ Peso (p.p.)": "{:+.2f}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    # ── Marginal contribution ────────────────────────────────────
    st.subheader("Contribuição marginal ao retorno esperado")
    marginal = bl_weights.reindex(asset_names).fillna(0) * bl_result.posterior_mu.reindex(asset_names).fillna(0)
    marginal_df = pd.DataFrame({
        "Ativo": [a.replace(".SA", "") for a in asset_names],
        "Contribuição": marginal.values,
    }).sort_values("Contribuição", ascending=False)
    marginal_df = marginal_df[marginal_df["Contribuição"].abs() > 1e-6]

    if not marginal_df.empty:
        fig2 = go.Figure(go.Bar(
            x=marginal_df["Ativo"], y=marginal_df["Contribuição"],
            marker_color="#4979f6",
        ))
        fig2.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#6B7280", size=11, family="Inter"),
            margin=dict(l=0, r=20, t=30, b=30),
            yaxis=dict(title="Contribuição ao retorno", tickformat=".2%", showgrid=True, gridcolor="#E5E7EB"),
            xaxis=dict(showgrid=False),
        )
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})
