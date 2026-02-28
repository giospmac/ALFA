from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from core.portfolio_repository import PortfolioRepository
from services.market_data import MarketDataError, fetch_ticker_comparison


DEFAULT_TICKERS = ["AAPL", "MSFT", "GOOGL", "NVDA", "AMZN", "TSLA", "META"]
HORIZON_OPTIONS = {
    "1 Month": "1mo",
    "3 Months": "3mo",
    "6 Months": "6mo",
    "1 Year": "1y",
    "5 Years": "5y",
    "10 Years": "10y",
    "20 Years": "20y",
}
CHART_COLORS = [
    "#8BC6FF",
    "#1E90FF",
    "#FFB3B3",
    "#FF3B30",
    "#82F2A4",
    "#27D3C3",
    "#FFD166",
    "#B794F4",
    "#F97316",
]


def _get_repository() -> PortfolioRepository:
    return PortfolioRepository(base_path=Path(__file__).resolve().parents[1])


def _deduplicate(items: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        normalized = str(item or "").strip().upper()
        if not normalized or normalized in seen:
            continue
        ordered.append(normalized)
        seen.add(normalized)
    return ordered


def _portfolio_tickers() -> list[str]:
    if "portfolio_df" in st.session_state:
        portfolio_df = st.session_state["portfolio_df"]
    else:
        portfolio_df = _get_repository().load_snapshot().portfolio

    if portfolio_df.empty or "ticker" not in portfolio_df.columns:
        return []

    equity_mask = portfolio_df.get("tipo_ativo", pd.Series(dtype=str)).astype(str).str.lower().eq("acao")
    tickers = portfolio_df.loc[equity_mask, "ticker"].astype(str).tolist() if equity_mask.any() else portfolio_df["ticker"].astype(str).tolist()
    return _deduplicate(tickers)


def _bootstrap_state() -> None:
    if "comparison_selected_tickers" not in st.session_state:
        st.session_state["comparison_selected_tickers"] = DEFAULT_TICKERS.copy()
    if "comparison_extra_tickers" not in st.session_state:
        st.session_state["comparison_extra_tickers"] = ""
    if "comparison_horizon" not in st.session_state:
        st.session_state["comparison_horizon"] = "6 Months"


def _render_styles() -> None:
    st.markdown(
        """
        <style>
        .alfa-card {
            background: linear-gradient(180deg, #182940 0%, #142235 100%);
            border: 1px solid rgba(142, 182, 255, 0.18);
            border-radius: 18px;
            padding: 1.25rem 1.25rem 1rem 1.25rem;
            min-height: 210px;
        }
        .alfa-card-label {
            color: #B8C7E0;
            font-size: 1rem;
            margin-bottom: 0.75rem;
        }
        .alfa-card-ticker {
            color: #EAF1FF;
            font-size: 2.2rem;
            line-height: 1;
            margin-bottom: 0.9rem;
        }
        .alfa-card-pill {
            display: inline-block;
            border-radius: 999px;
            padding: 0.25rem 0.7rem;
            font-size: 0.95rem;
            font-weight: 600;
        }
        .alfa-card-pill.positive {
            background: rgba(56, 221, 126, 0.15);
            color: #5FE58F;
        }
        .alfa-card-pill.negative {
            background: rgba(255, 91, 91, 0.16);
            color: #FF8E8E;
        }
        div[data-testid="stButton"] > button {
            border-radius: 999px;
            min-height: 2.75rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _parse_extra_tickers(raw_value: str) -> list[str]:
    if not raw_value.strip():
        return []
    tokens = [token.strip() for token in raw_value.replace(";", ",").split(",")]
    return _deduplicate(tokens)


def _build_selection() -> tuple[list[str], list[str]]:
    option_pool = _deduplicate(DEFAULT_TICKERS + _portfolio_tickers())
    extra_tickers = _parse_extra_tickers(st.session_state["comparison_extra_tickers"])
    selected_tickers = _deduplicate(st.session_state["comparison_selected_tickers"] + extra_tickers)
    return option_pool, selected_tickers


def _render_horizon_selector() -> str:
    st.markdown("##### Time horizon")
    horizon_rows = [
        ["1 Month", "3 Months", "6 Months"],
        ["1 Year", "5 Years", "10 Years", "20 Years"],
    ]
    for row in horizon_rows:
        columns = st.columns(len(row))
        for column, label in zip(columns, row):
            button_type = "primary" if st.session_state["comparison_horizon"] == label else "secondary"
            if column.button(label, key=f"horizon-{label}", use_container_width=True, type=button_type):
                st.session_state["comparison_horizon"] = label
    return st.session_state["comparison_horizon"]


def _performance_card(title: str, ticker: str, total_return: float) -> str:
    direction = "positive" if total_return >= 0 else "negative"
    arrow = "&uarr;" if total_return >= 0 else "&darr;"
    return (
        f"<div class='alfa-card'>"
        f"<div class='alfa-card-label'>{title}</div>"
        f"<div class='alfa-card-ticker'>{ticker}</div>"
        f"<div class='alfa-card-pill {direction}'>{arrow} {abs(total_return) * 100:.0f}%</div>"
        f"</div>"
    )


def _plot_history(history: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor("#15263D")
    ax.set_facecolor("#15263D")

    for index, column in enumerate(history.columns):
        ax.plot(
            history.index,
            history[column],
            linewidth=2.2,
            color=CHART_COLORS[index % len(CHART_COLORS)],
            label=column,
        )

    ax.set_ylabel("Normalized price", color="#D8E3F5", fontsize=12)
    ax.set_xlabel("Date", color="#D8E3F5", fontsize=12, labelpad=14)
    ax.grid(True, axis="y", color="#314760", alpha=0.75, linewidth=0.8)
    ax.grid(False, axis="x")
    ax.tick_params(axis="x", colors="#D8E3F5", labelrotation=0)
    ax.tick_params(axis="y", colors="#D8E3F5")
    for spine in ax.spines.values():
        spine.set_visible(False)

    legend = ax.legend(
        title="Stock",
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        frameon=False,
        labelcolor="#EAF1FF",
        borderaxespad=0,
        handlelength=0.8,
        handletextpad=0.5,
    )
    if legend is not None:
        legend.get_title().set_color("#EAF1FF")

    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def render_stock_comparison_page() -> None:
    _bootstrap_state()
    _render_styles()

    st.title("Comparador de Ações")
    st.caption("Compare a performance relativa de múltiplos tickers no período escolhido.")

    option_pool, selected_tickers = _build_selection()

    left_col, right_col = st.columns([1.1, 3.2], gap="large")

    with left_col:
        with st.container(border=True):
            st.markdown("##### Stock tickers")
            st.multiselect(
                "Tickers",
                options=option_pool,
                key="comparison_selected_tickers",
                label_visibility="collapsed",
                placeholder="Selecione os tickers",
            )
            st.text_input(
                "Adicionar tickers",
                key="comparison_extra_tickers",
                placeholder="Ex.: PETR4, VALE3.SA, ASML",
            )

        horizon_label = _render_horizon_selector()

    if not selected_tickers:
        with right_col:
            st.info("Selecione pelo menos um ticker para exibir a comparação.")
        return

    if len(selected_tickers) > 9:
        selected_tickers = selected_tickers[:9]
        st.warning("A visualização foi limitada aos 9 primeiros tickers para manter a leitura do gráfico.")

    try:
        with st.spinner("Consultando séries históricas..."):
            result = fetch_ticker_comparison(tuple(selected_tickers), HORIZON_OPTIONS[horizon_label])
    except MarketDataError as exc:
        with right_col:
            st.error(str(exc))
        return
    except Exception:
        with right_col:
            st.error("Ocorreu um erro inesperado ao montar a comparação.")
        return

    if result.invalid_tickers:
        st.warning("Tickers ignorados por falta de histórico suficiente: " + ", ".join(result.invalid_tickers))

    if not result.total_returns:
        with right_col:
            st.info("Não há dados suficientes para comparar os tickers selecionados.")
        return

    sorted_returns = sorted(result.total_returns.items(), key=lambda item: item[1], reverse=True)
    best_ticker, best_return = sorted_returns[0]
    worst_ticker, worst_return = sorted_returns[-1]

    with left_col:
        summary_col_1, summary_col_2 = st.columns(2)
        summary_col_1.markdown(_performance_card("Best stock", best_ticker, best_return), unsafe_allow_html=True)
        summary_col_2.markdown(_performance_card("Worst stock", worst_ticker, worst_return), unsafe_allow_html=True)

    with right_col:
        with st.container(border=True):
            history = result.normalized_history.copy()
            history.index = pd.to_datetime(history.index, errors="coerce")
            history = history[history.index.notna()]
            _plot_history(history)
            if not history.empty:
                start_date = history.index.min().strftime("%d/%m/%Y")
                end_date = history.index.max().strftime("%d/%m/%Y")
                st.caption(f"Janela analisada: {start_date} a {end_date}. Base normalizada em 1.0 no primeiro ponto válido.")
