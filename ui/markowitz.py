from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.portfolio_repository import PortfolioRepository
from services.portfolio_analytics import markowitz_frontier


def _load_data():
    if "portfolio_df" in st.session_state and "historical_df" in st.session_state:
        return st.session_state["portfolio_df"], st.session_state["historical_df"]

    repo = PortfolioRepository(base_path=Path(__file__).resolve().parents[1])
    snapshot = repo.load_snapshot()
    return snapshot.portfolio, snapshot.historical


def render_markowitz_page() -> None:
    portfolio_df, historical_df = _load_data()

    st.title("Fronteira Eficiente")
    st.caption("Simulação de carteiras para destacar máximo Sharpe, mínima volatilidade e a fronteira eficiente.")

    if portfolio_df.empty or historical_df.empty:
        st.info("Monte o portfolio e atualize o histórico na página principal para liberar a análise de Markowitz.")
        return

    result = markowitz_frontier(portfolio_df, historical_df)
    if result is None:
        st.info("São necessários pelo menos dois ativos válidos e cerca de 21 observações para a simulação.")
        return

    custom_blues = [[0.0, "#93c5fd"], [0.35, "#4979f6"], [0.7, "#1e379b"], [1.0, "#102170"]]

    fig = go.Figure()

    # Monte Carlo scatter cloud
    fig.add_trace(go.Scatter(
        x=result.portfolios["volatilidade"],
        y=result.portfolios["retorno"],
        mode="markers",
        marker=dict(
            size=3.5,
            color=result.portfolios["sharpe"],
            colorscale=custom_blues,
            showscale=True,
            colorbar=dict(title="Sharpe", title_side="right", thickness=12, len=0.6),
            opacity=0.45,
        ),
        name="Portfólios simulados",
        hovertemplate="Vol: %{x:.2%}<br>Ret: %{y:.2%}<br>Sharpe: %{marker.color:.2f}<extra></extra>",
    ))

    # Efficient frontier: split into upper (efficient) and lower (inefficient)
    if not result.frontier.empty:
        min_vol_ret = result.min_vol_return
        upper = result.frontier[result.frontier["retorno"] >= min_vol_ret - 1e-6]
        lower = result.frontier[result.frontier["retorno"] <= min_vol_ret + 1e-6]

        if not upper.empty:
            fig.add_trace(go.Scatter(
                x=upper["volatilidade"],
                y=upper["retorno"],
                mode="lines",
                line=dict(color="#1e379b", width=2.5),
                name="Fronteira eficiente",
                hoverinfo="skip",
            ))
        if not lower.empty:
            fig.add_trace(go.Scatter(
                x=lower["volatilidade"],
                y=lower["retorno"],
                mode="lines",
                line=dict(color="#93c5fd", width=1.5, dash="dash"),
                name="Fronteira ineficiente",
                hoverinfo="skip",
            ))

    # Max Sharpe point
    fig.add_trace(go.Scatter(
        x=[result.max_sharpe_volatility],
        y=[result.max_sharpe_return],
        mode="markers",
        marker=dict(size=14, symbol="star", color="#4979f6", line=dict(color="#1e379b", width=1.5)),
        name="Máximo Sharpe",
        hovertemplate="Vol: %{x:.2%}<br>Ret: %{y:.2%}<extra></extra>",
    ))

    # Min volatility point
    fig.add_trace(go.Scatter(
        x=[result.min_vol_volatility],
        y=[result.min_vol_return],
        mode="markers",
        marker=dict(size=10, symbol="diamond", color="#1e379b", line=dict(color="#102170", width=1.5)),
        name="Mínima volatilidade",
        hovertemplate="Vol: %{x:.2%}<br>Ret: %{y:.2%}<extra></extra>",
    ))

    # Axis ranges: include cloud + frontier + special points with padding
    all_vols = pd.concat([
        result.portfolios["volatilidade"],
        result.frontier["volatilidade"],
        pd.Series([result.max_sharpe_volatility, result.min_vol_volatility]),
    ])
    all_rets = pd.concat([
        result.portfolios["retorno"],
        result.frontier["retorno"],
        pd.Series([result.max_sharpe_return, result.min_vol_return]),
    ])
    pad_x = (all_vols.max() - all_vols.min()) * 0.06
    pad_y = (all_rets.max() - all_rets.min()) * 0.06

    fig.update_layout(
        title=dict(
            text="Fronteira Eficiente de Markowitz",
            font=dict(color="#111827", size=14, family="Inter"),
            pad=dict(b=10),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#6B7280", size=11, family="Inter"),
        margin=dict(l=0, r=20, t=60, b=40),
        hovermode="closest",
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="right", x=1, title="",
        ),
    )
    fig.update_xaxes(
        title_text="Volatilidade anualizada",
        showgrid=True, gridcolor="#E5E7EB", gridwidth=1,
        zeroline=False, showline=True, linecolor="#E5E7EB", linewidth=1,
        tickformat=".0%",
        range=[all_vols.min() - pad_x, all_vols.max() + pad_x],
    )
    fig.update_yaxes(
        title_text="Retorno anualizado esperado",
        showgrid=True, gridcolor="#E5E7EB", gridwidth=1,
        zeroline=False, showline=False,
        tickformat=".0%",
        range=[all_rets.min() - pad_y, all_rets.max() + pad_y],
    )

    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # Optimized weights table
    st.subheader("Carteiras otimizadas")
    weights_df = pd.DataFrame({
        "Ativo": result.max_sharpe_weights.index.str.replace(".SA", "", regex=False),
        "Max Sharpe (%)": result.max_sharpe_weights.values * 100,
        "Mín. Volatilidade (%)": result.min_vol_weights.reindex(result.max_sharpe_weights.index).values * 100,
    })
    # Only show assets with meaningful allocation
    weights_df = weights_df[
        (weights_df["Max Sharpe (%)"] > 0.1) | (weights_df["Mín. Volatilidade (%)"] > 0.1)
    ].sort_values("Max Sharpe (%)", ascending=False)

    st.dataframe(
        weights_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Max Sharpe (%)": st.column_config.NumberColumn(format="%.2f"),
            "Mín. Volatilidade (%)": st.column_config.NumberColumn(format="%.2f"),
        },
    )

    # Summary metrics
    col1, col2 = st.columns(2)
    col1.metric("Max Sharpe", f"{result.max_sharpe_return:.1%} ret · {result.max_sharpe_volatility:.1%} vol")
    col2.metric("Mín. Volatilidade", f"{result.min_vol_return:.1%} ret · {result.min_vol_volatility:.1%} vol")
