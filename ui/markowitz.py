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
    st.caption("Simulacao de carteiras para destacar maximo Sharpe, minima volatilidade e a fronteira eficiente.")

    if portfolio_df.empty or historical_df.empty:
        st.info("Monte o portfolio e atualize o historico na pagina principal para liberar a analise de Markowitz.")
        return

    result = markowitz_frontier(portfolio_df, historical_df)
    if result is None:
        st.info("Sao necessarios pelo menos dois ativos validos e cerca de 21 observacoes para a simulacao.")
        return

    blue_map = LinearSegmentedColormap.from_list(
        "alfa_blues",
        ["#97bdff", "#4979f6", "#1e379b", "#102170", "#020838"],
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    scatter = ax.scatter(
        result.portfolios["volatilidade"],
        result.portfolios["retorno"],
        c=result.portfolios["sharpe"],
        cmap=blue_map,
        s=10,
        alpha=0.6,
    )
    ax.scatter(
        result.max_sharpe_volatility,
        result.max_sharpe_return,
        s=160,
        marker="*",
        edgecolors="black",
        label="Maximo Sharpe",
    )
    ax.scatter(
        result.min_vol_volatility,
        result.min_vol_return,
        s=90,
        marker="D",
        edgecolors="black",
        label="Minima volatilidade",
    )
    ax.plot(result.frontier["volatilidade"], result.frontier["retorno"], linestyle="--", linewidth=2, label="Fronteira eficiente")
    ax.set_xlabel("Volatilidade anualizada")
    ax.set_ylabel("Retorno anualizado esperado")
    ax.set_title("Fronteira Eficiente de Markowitz")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="upper left")
    fig.colorbar(scatter, ax=ax, label="Indice de Sharpe")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.subheader("Carteiras otimizadas via simulacao")
    weights_df = pd.DataFrame(
        {
            "Ativo": result.max_sharpe_weights.index,
            "Max Sharpe (%)": result.max_sharpe_weights.values * 100,
            "Minima Volatilidade (%)": result.min_vol_weights.reindex(result.max_sharpe_weights.index).values * 100,
        }
    )
    st.dataframe(
        weights_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Max Sharpe (%)": st.column_config.NumberColumn(format="%.2f"),
            "Minima Volatilidade (%)": st.column_config.NumberColumn(format="%.2f"),
        },
    )
