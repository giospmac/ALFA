from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import streamlit as st

from core.portfolio_repository import PortfolioRepository
from services.portfolio_analytics import (
    benchmark_return_series,
    contribution_by_asset,
    drawdown_series,
    monte_carlo_simulation,
    rolling_volatility_series,
)

PALETTE = ["#2563EB", "#16A34A", "#4979f6", "#D97706", "#7C3AED"]


def _load_data():
    if "portfolio_df" in st.session_state and "historical_df" in st.session_state:
        return st.session_state["portfolio_df"], st.session_state["historical_df"]

    repo = PortfolioRepository(base_path=Path(__file__).resolve().parents[1])
    snapshot = repo.load_snapshot()
    return snapshot.portfolio, snapshot.historical


def _clean_ax(ax, fig) -> None:
    """Apply a consistent clean style to a matplotlib axes."""
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")
    ax.grid(True, axis="y", color="#F3F4F6", linewidth=0.9, linestyle="-")
    ax.grid(False, axis="x")
    ax.tick_params(axis="both", colors="#6B7280", labelsize=9)
    ax.title.set_color("#111827")
    ax.title.set_fontsize(11)
    ax.title.set_fontweight("600")
    for spine in ax.spines.values():
        spine.set_visible(False)


def _plot_return_comparison(years: int | None, months: int | None, title: str, benchmark: str):
    portfolio_df, historical_df = _load_data()
    chart_df = benchmark_return_series(portfolio_df, historical_df, benchmark, years=years, months=months)
    if chart_df.empty:
        st.info(f"Dados insuficientes para {title}.")
        return

    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    ax.plot(chart_df.index, chart_df["Portfolio"], label="Portfolio", linewidth=2, color=PALETTE[0])
    ax.plot(chart_df.index, chart_df[benchmark], label=benchmark, linewidth=2, color=PALETTE[1])
    ax.set_title(title)
    ax.set_ylabel("Retorno acumulado (%)", color="#6B7280", fontsize=9)
    ax.legend(fontsize=8, frameon=False, labelcolor="#374151")
    _clean_ax(ax, fig)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


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
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(contribution_df["ticker"], contribution_df["contribuicao"] * 100, color=PALETTE[0], width=0.55)
        ax.set_ylabel("Contribuição (%)", color="#6B7280", fontsize=9)
        ax.set_title("Contribuição de cada ativo no último ano")
        _clean_ax(ax, fig)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    st.subheader("Drawdown histórico")
    drawdown = drawdown_series(portfolio_df, historical_df)
    if drawdown.empty:
        st.info("Dados insuficientes para calcular o drawdown do portfolio.")
    else:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.fill_between(drawdown.index, drawdown.values, 0, alpha=0.12, color=PALETTE[2])
        ax.plot(drawdown.index, drawdown.values, linewidth=1.5, color=PALETTE[2])
        ax.set_ylabel("Drawdown (%)", color="#6B7280", fontsize=9)
        ax.set_title("Histórico de drawdown do portfolio")
        _clean_ax(ax, fig)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    st.subheader("Simulação de Monte Carlo")
    monte_carlo = monte_carlo_simulation(portfolio_df, historical_df)
    if monte_carlo is None:
        st.info("Dados insuficientes para rodar a simulação de Monte Carlo.")
    else:
        metric_col_1, metric_col_2 = st.columns(2)
        metric_col_1.metric("VaR 5%", f"{monte_carlo.var_5:.2f}%")
        metric_col_2.metric("CVaR 5%", f"{monte_carlo.cvar_5:.2f}%")

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.hist(monte_carlo.simulated_returns * 100, bins=50, density=True, alpha=0.75, color=PALETTE[0])
        ax.axvline(monte_carlo.var_5, linestyle="--", linewidth=1.8, color=PALETTE[2], label=f"VaR 5%: {monte_carlo.var_5:.2f}%")
        ax.axvline(monte_carlo.cvar_5, linestyle="--", linewidth=1.8, color=PALETTE[3], label=f"CVaR 5%: {monte_carlo.cvar_5:.2f}%")
        ax.set_title("Distribuição dos retornos simulados em 4 semanas")
        ax.set_xlabel("Retorno (%)", color="#6B7280", fontsize=9)
        ax.legend(fontsize=8, frameon=False, labelcolor="#374151")
        _clean_ax(ax, fig)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    st.subheader("Volatilidade anualizada rolling")
    rolling_vol = rolling_volatility_series(portfolio_df, historical_df)
    if rolling_vol.empty:
        st.info("Dados insuficientes para calcular a volatilidade rolling de 21 dias.")
    else:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(rolling_vol.index, rolling_vol.values, linewidth=2, color=PALETTE[0])
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.set_ylabel("Volatilidade", color="#6B7280", fontsize=9)
        ax.set_title("Volatilidade anualizada · janela de 21 dias úteis")
        _clean_ax(ax, fig)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
