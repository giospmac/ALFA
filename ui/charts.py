from __future__ import annotations

from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

from core.portfolio_repository import PortfolioRepository
from services.portfolio_analytics import (
    benchmark_return_series,
    contribution_by_asset,
    drawdown_series,
    monte_carlo_simulation,
    rolling_volatility_series,
)

PALETTE_PRIMARY = "#4979f6"
PALETTE_SECONDARY = "#059669"
PALETTE_NEGATIVE = "#EF4444"
PALETTE_WARNING = "#D97706"


def _load_data():
    if "portfolio_df" in st.session_state and "historical_df" in st.session_state:
        return st.session_state["portfolio_df"], st.session_state["historical_df"]

    repo = PortfolioRepository(base_path=Path(__file__).resolve().parents[1])
    snapshot = repo.load_snapshot()
    return snapshot.portfolio, snapshot.historical


def _apply_alfa_style(fig: go.Figure, title: str = "") -> go.Figure:
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(color="#111827", size=14, family="Inter"),
            pad=dict(b=10)
        ),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(color="#6B7280", size=11, family="Inter"),
        margin=dict(l=40, r=20, t=60, b=40),
        hovermode="x unified",
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
        showgrid=False,
        zeroline=False,
        showline=True,
        linecolor="#E5E7EB",
        linewidth=1
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="#E5E7EB",
        gridwidth=1,
        zeroline=False,
        showline=False
    )
    return fig


def _plot_return_comparison(years: int | None, months: int | None, title: str, benchmark: str):
    portfolio_df, historical_df = _load_data()
    chart_df = benchmark_return_series(portfolio_df, historical_df, benchmark, years=years, months=months)
    if chart_df.empty:
        st.info(f"Dados insuficientes para {title}.")
        return

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df["Portfolio"], mode="lines", name="Portfolio", line=dict(color=PALETTE_PRIMARY, width=2)))
    fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df[benchmark], mode="lines", name=benchmark, line=dict(color=PALETTE_SECONDARY, width=2)))
    
    _apply_alfa_style(fig, title=title)
    fig.update_yaxes(title_text="Retorno acumulado (%)")
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})


def render_charts_page() -> None:
    portfolio_df, historical_df = _load_data()

    st.title("Histórico")
    st.caption("Comparações com benchmark, contribuição por ativo, drawdown, Monte Carlo e volatilidade rolling.")

    if portfolio_df.empty or historical_df.empty:
        st.info("Monte o portfolio e atualize o histórico na página principal para liberar os gráficos.")
        return

    st.subheader("Retorno acumulado · Portfolio vs Ibovespa")
    ibov_col_1, ibov_col_2, ibov_col_3 = st.columns(3)
    with ibov_col_1:
        _plot_return_comparison(None, 1, "1 mês", "IBOVESPA")
    with ibov_col_2:
        _plot_return_comparison(1, None, "1 ano", "IBOVESPA")
    with ibov_col_3:
        _plot_return_comparison(5, None, "5 anos", "IBOVESPA")

    st.subheader("Retorno acumulado · Portfolio vs CDI")
    cdi_col_1, cdi_col_2, cdi_col_3 = st.columns(3)
    with cdi_col_1:
        _plot_return_comparison(None, 1, "1 mês", "CDI")
    with cdi_col_2:
        _plot_return_comparison(1, None, "1 ano", "CDI")
    with cdi_col_3:
        _plot_return_comparison(5, None, "5 anos", "CDI")

    st.subheader("Contribuição de retorno por ativo")
    contribution_df = contribution_by_asset(portfolio_df, historical_df)
    if contribution_df.empty:
        st.info("Dados insuficientes para calcular a contribuição de retorno dos ativos.")
    else:
        fig = go.Figure(data=[
            go.Bar(
                x=contribution_df["ticker"], 
                y=contribution_df["contribuicao"] * 100,
                marker_color=PALETTE_PRIMARY,
                width=0.4
            )
        ])
        _apply_alfa_style(fig, title="Contribuição de cada ativo no último ano")
        fig.update_yaxes(title_text="Contribuição (%)")
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    st.subheader("Drawdown histórico")
    drawdown = drawdown_series(portfolio_df, historical_df)
    if drawdown.empty:
        st.info("Dados insuficientes para calcular o drawdown do portfolio.")
    else:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=drawdown.index, 
            y=drawdown.values, 
            fill='tozeroy', 
            mode='lines',
            line=dict(color=PALETTE_PRIMARY, width=1.5),
            fillcolor='rgba(73, 121, 246, 0.15)',
            name="Drawdown"
        ))
        _apply_alfa_style(fig, title="Histórico de drawdown do portfolio")
        fig.update_yaxes(title_text="Drawdown (%)")
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    st.subheader("Simulação de Monte Carlo")
    monte_carlo = monte_carlo_simulation(portfolio_df, historical_df)
    if monte_carlo is None:
        st.info("Dados insuficientes para rodar a simulação de Monte Carlo.")
    else:
        metric_col_1, metric_col_2 = st.columns(2)
        metric_col_1.metric("VaR 5%", f"{monte_carlo.var_5:.2f}%")
        metric_col_2.metric("CVaR 5%", f"{monte_carlo.cvar_5:.2f}%")

        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=monte_carlo.simulated_returns * 100, 
            nbinsx=50, 
            histnorm='probability density',
            marker_color=PALETTE_PRIMARY,
            opacity=0.75,
            name="Retornos"
        ))
        
        # Add vertical lines via shapes and annotations for Plotly
        fig.add_vline(x=monte_carlo.var_5, line_dash="dash", line_color=PALETTE_NEGATIVE, annotation_text=f"VaR 5%: {monte_carlo.var_5:.2f}%", annotation_position="top left")
        fig.add_vline(x=monte_carlo.cvar_5, line_dash="dash", line_color=PALETTE_WARNING, annotation_text=f"CVaR 5%: {monte_carlo.cvar_5:.2f}%", annotation_position="top right")
        
        _apply_alfa_style(fig, title="Distribuição dos retornos simulados em 4 semanas")
        fig.update_xaxes(title_text="Retorno (%)")
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    st.subheader("Volatilidade anualizada rolling")
    rolling_vol = rolling_volatility_series(portfolio_df, historical_df)
    if rolling_vol.empty:
        st.info("Dados insuficientes para calcular a volatilidade rolling de 21 dias.")
    else:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=rolling_vol.index, y=rolling_vol.values, mode='lines', line=dict(color=PALETTE_PRIMARY, width=2), name="Volatilidade"))
        
        _apply_alfa_style(fig, title="Volatilidade anualizada · janela de 21 dias úteis")
        fig.update_yaxes(title_text="Volatilidade", tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
