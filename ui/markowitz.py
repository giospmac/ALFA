from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
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

    blue_map = LinearSegmentedColormap.from_list(
        "alfa_blues",
        ["#BFDBFE", "#3B82F6", "#1D4ED8", "#1E3A8A"],
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")

    scatter = ax.scatter(
        result.portfolios["volatilidade"],
        result.portfolios["retorno"],
        c=result.portfolios["sharpe"],
        cmap=blue_map,
        s=8,
        alpha=0.5,
    )
    ax.scatter(
        result.max_sharpe_volatility,
        result.max_sharpe_return,
        s=140,
        marker="*",
        color="#2563EB",
        edgecolors="#1D4ED8",
        linewidths=0.8,
        zorder=5,
        label="Máximo Sharpe",
    )
    ax.scatter(
        result.min_vol_volatility,
        result.min_vol_return,
        s=80,
        marker="D",
        color="#16A34A",
        edgecolors="#15803D",
        linewidths=0.8,
        zorder=5,
        label="Mínima volatilidade",
    )
    ax.plot(
        result.frontier["volatilidade"],
        result.frontier["retorno"],
        linestyle="--",
        linewidth=1.8,
        color="#6B7280",
        label="Fronteira eficiente",
    )
    ax.set_xlabel("Volatilidade anualizada", color="#6B7280", fontsize=10)
    ax.set_ylabel("Retorno anualizado esperado", color="#6B7280", fontsize=10)
    ax.set_title("Fronteira Eficiente de Markowitz", color="#111827", fontsize=12, fontweight="600")
    ax.grid(True, axis="both", color="#F3F4F6", linewidth=0.9)
    ax.tick_params(colors="#6B7280", labelsize=9)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.legend(loc="upper left", frameon=False, labelcolor="#374151", fontsize=9)
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Índice de Sharpe", color="#6B7280", fontsize=9)
    cbar.ax.tick_params(colors="#6B7280", labelsize=8)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

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

