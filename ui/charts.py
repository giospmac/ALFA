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
PALETTE_SECONDARY = "#1e379b"
PALETTE_NEGATIVE = "#1e3a8a"
PALETTE_WARNING = "#60a5fa"


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
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#6B7280", size=11, family="Inter"),
        margin=dict(l=0, r=20, t=60, b=40),
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
    
    _apply_alfa_style(fig)
    fig.update_yaxes(title_text="")
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})


def render_charts_page() -> None:
    portfolio_df, historical_df = _load_data()

    st.title("Histórico")
    st.caption("Comparações com benchmark, contribuição por ativo, drawdown, Monte Carlo e volatilidade rolling.")

    if portfolio_df.empty or historical_df.empty:
        st.info("Monte o portfolio e atualize o histórico na página principal para liberar os gráficos.")
        return

    st.subheader("Retorno acumulado · Portfolio vs Ibovespa")
    period_options = {
        "1 mês": (None, 1),
        "3 meses": (None, 3),
        "6 meses": (None, 6),
        "1 ano": (1, None),
        "2 anos": (2, None),
        "5 anos": (5, None),
    }
    
    val_ibov = st.pills("Período (Ibovespa)", options=list(period_options.keys()), default="5 anos", key="period_ibov", label_visibility="collapsed")
    if val_ibov:
        y_ibov, m_ibov = period_options[val_ibov]
        _plot_return_comparison(y_ibov, m_ibov, "IBOVESPA", "IBOVESPA")

    st.subheader("Retorno acumulado · Portfolio vs CDI")
    val_cdi = st.pills("Período (CDI)", options=list(period_options.keys()), default="5 anos", key="period_cdi", label_visibility="collapsed")
    if val_cdi:
        y_cdi, m_cdi = period_options[val_cdi]
        _plot_return_comparison(y_cdi, m_cdi, "CDI", "CDI")

    st.subheader("Contribuição de retorno por ativo")
    contribution_df = contribution_by_asset(portfolio_df, historical_df)
    if contribution_df.empty:
        st.info("Dados insuficientes para calcular a contribuição de retorno dos ativos.")
    else:
        x_vals = contribution_df["ticker"].tolist() + ["Total"]
        y_vals = (contribution_df["contribuicao"] * 100).tolist() + [0]
        measure_vals = ["relative"] * len(contribution_df) + ["total"]

        fig = go.Figure(data=[
            go.Waterfall(
                x=x_vals,
                y=y_vals,
                measure=measure_vals,
                decreasing=dict(marker=dict(color="#EF4444")),
                increasing=dict(marker=dict(color=PALETTE_PRIMARY)),
                totals=dict(marker=dict(color="#1e379b")),
                width=0.5,
                textposition="outside",
                texttemplate="%{y:.2f}%"
            )
        ])
        _apply_alfa_style(fig, title="")
        fig.update_yaxes(title_text="", showticklabels=False, showgrid=True, gridcolor="#E5E7EB", zeroline=True)
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
        _apply_alfa_style(fig)
        fig.update_yaxes(title_text="", tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    st.subheader("Simulação de Monte Carlo")
    monte_carlo = monte_carlo_simulation(portfolio_df, historical_df)
    if monte_carlo is None:
        st.info("Dados insuficientes para rodar a simulação de Monte Carlo.")
    else:
        metric_col_1, metric_col_2 = st.columns(2)
        metric_col_1.metric("VaR 5%", f"{monte_carlo.var_5:.2f}%")
        metric_col_2.metric("CVaR 5%", f"{monte_carlo.cvar_5:.2f}%")

        import numpy as np
        from scipy.stats import gaussian_kde

        sim_pct = monte_carlo.simulated_returns * 100
        kde = gaussian_kde(sim_pct, bw_method=0.25)
        x_range = np.linspace(float(sim_pct.min()) - 2, float(sim_pct.max()) + 2, 300)
        y_kde = kde(x_range)

        fig = go.Figure()

        # Tail region (below VaR) — shaded red
        tail_mask = x_range <= monte_carlo.var_5
        if tail_mask.any():
            fig.add_trace(go.Scatter(
                x=x_range[tail_mask], y=y_kde[tail_mask],
                fill="tozeroy", mode="lines",
                line=dict(width=0),
                fillcolor="rgba(30, 55, 155, 0.25)",
                name="Cauda (VaR 5%)",
                hoverinfo="skip",
            ))

        # Full distribution curve — main fill
        fig.add_trace(go.Scatter(
            x=x_range, y=y_kde,
            fill="tozeroy", mode="lines",
            line=dict(color="#4979f6", width=2.2),
            fillcolor="rgba(73, 121, 246, 0.12)",
            name="Distribuição",
            hovertemplate="Retorno: %{x:.1f}%<br>Densidade: %{y:.4f}<extra></extra>",
        ))

        # VaR line
        fig.add_trace(go.Scatter(
            x=[monte_carlo.var_5, monte_carlo.var_5],
            y=[0, float(kde([monte_carlo.var_5])[0])],
            mode="lines",
            line=dict(color="#1e379b", width=2, dash="dash"),
            name=f"VaR 5%  ({monte_carlo.var_5:.2f}%)",
            hoverinfo="skip",
        ))

        # CVaR line
        fig.add_trace(go.Scatter(
            x=[monte_carlo.cvar_5, monte_carlo.cvar_5],
            y=[0, float(kde([monte_carlo.cvar_5])[0])],
            mode="lines",
            line=dict(color="#102263", width=2, dash="dot"),
            name=f"CVaR 5%  ({monte_carlo.cvar_5:.2f}%)",
            hoverinfo="skip",
        ))

        _apply_alfa_style(fig)
        fig.update_layout(
            showlegend=True,
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                font=dict(size=10),
            ),
        )
        fig.update_xaxes(title_text="", ticksuffix="%")
        fig.update_yaxes(showticklabels=False, showgrid=False, title_text="")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    st.subheader("Volatilidade anualizada rolling")
    rolling_vol = rolling_volatility_series(portfolio_df, historical_df)
    if rolling_vol.empty:
        st.info("Dados insuficientes para calcular a volatilidade rolling de 21 dias.")
    else:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=rolling_vol.index, y=rolling_vol.values, mode='lines', line=dict(color=PALETTE_PRIMARY, width=2), name="Volatilidade"))
        
        _apply_alfa_style(fig)
        fig.update_yaxes(title_text="", tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
