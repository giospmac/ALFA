from __future__ import annotations

from pathlib import Path

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

    custom_blues = [[0.0, "#4979f6"], [0.5, "#1e379b"], [1.0, "#102170"]]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=result.portfolios["volatilidade"],
        y=result.portfolios["retorno"],
        mode="markers",
        marker=dict(
            size=5,
            color=result.portfolios["sharpe"],
            colorscale=custom_blues,
            showscale=True,
            colorbar=dict(title="Índice de Sharpe", title_side="right", thickness=15),
            opacity=0.5
        ),
        name="Portfólios simulados",
        hovertemplate="Volatilidade: %{x:.2%}<br>Retorno: %{y:.2%}<br>Sharpe: %{marker.color:.2f}<extra></extra>"
    ))

    frontier_sorted = result.frontier.sort_values("volatilidade")

    fig.add_trace(go.Scatter(
        x=frontier_sorted["volatilidade"],
        y=frontier_sorted["retorno"],
        mode="lines",
        line=dict(color="#1e379b", width=2, shape="spline"),
        name="Fronteira eficiente",
        hoverinfo="skip"
    ))

    fig.add_trace(go.Scatter(
        x=[result.max_sharpe_volatility],
        y=[result.max_sharpe_return],
        mode="markers",
        marker=dict(size=14, symbol="star", color="#4979f6", line=dict(color="#1e379b", width=1)),
        name="Máximo Sharpe",
        hovertemplate="Volatilidade: %{x:.2%}<br>Retorno: %{y:.2%}<extra></extra>"
    ))

    fig.add_trace(go.Scatter(
        x=[result.min_vol_volatility],
        y=[result.min_vol_return],
        mode="markers",
        marker=dict(size=10, symbol="diamond", color="#1e379b", line=dict(color="#102170", width=1)),
        name="Mínima volatilidade",
        hovertemplate="Volatilidade: %{x:.2%}<br>Retorno: %{y:.2%}<extra></extra>"
    ))

    fig.update_layout(
        title=dict(
            text="Fronteira Eficiente de Markowitz",
            font=dict(color="#111827", size=14, family="Inter"),
            pad=dict(b=10)
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#6B7280", size=11, family="Inter"),
        margin=dict(l=0, r=20, t=60, b=40),
        hovermode="closest",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            title=""
        )
    )
    fig.update_xaxes(
        title_text="Volatilidade anualizada",
        showgrid=True,
        gridcolor="#E5E7EB",
        gridwidth=1,
        zeroline=False,
        showline=True,
        linecolor="#E5E7EB",
        linewidth=1
    )
    fig.update_yaxes(
        title_text="Retorno anualizado esperado",
        showgrid=True,
        gridcolor="#E5E7EB",
        gridwidth=1,
        zeroline=False,
        showline=False
    )

    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    st.subheader("Carteiras otimizadas via simulação")
    weights_df = pd.DataFrame(
        {
            "Ativo": result.max_sharpe_weights.index,
            "Max Sharpe (%)": result.max_sharpe_weights.values * 100,
            "Mínima Volatilidade (%)": result.min_vol_weights.reindex(result.max_sharpe_weights.index).values * 100,
        }
    )
    st.dataframe(
        weights_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Max Sharpe (%)": st.column_config.NumberColumn(format="%.2f"),
            "Mínima Volatilidade (%)": st.column_config.NumberColumn(format="%.2f"),
        },
    )

