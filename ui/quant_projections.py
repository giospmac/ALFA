from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from core.portfolio_repository import PortfolioRepository
from services.market_data import MarketDataError, fetch_ticker_quant_projection


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
    if "quant_selected_tickers" not in st.session_state:
        st.session_state["quant_selected_tickers"] = DEFAULT_TICKERS.copy()
    if "quant_extra_tickers" not in st.session_state:
        st.session_state["quant_extra_tickers"] = ""
    if "quant_horizon" not in st.session_state:
        st.session_state["quant_horizon"] = "6 Months"


def _render_styles() -> None:
    st.markdown(
        """
        <style>
        .alfa-card {
            background: #FFFFFF;
            border: 1px solid #E5E7EB;
            border-radius: 10px;
            padding: 1.1rem 1.2rem;
        }
        .alfa-card-label {
            color: #6B7280;
            font-size: 0.73rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.07em;
            margin-bottom: 0.55rem;
        }
        .alfa-card-ticker {
            color: #111827;
            font-size: 1.7rem;
            font-weight: 700;
            line-height: 1.1;
            margin-bottom: 0.65rem;
            letter-spacing: -0.02em;
        }
        .alfa-card-pill {
            display: inline-block;
            border-radius: 6px;
            padding: 0.2rem 0.55rem;
            font-size: 0.85rem;
            font-weight: 600;
        }
        .alfa-card-pill.positive {
            background: #ECFDF5;
            color: #059669;
        }
        .alfa-card-pill.negative {
            background: #EFF6FF;
            color: #4979f6;
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
    extra_tickers = _parse_extra_tickers(st.session_state["quant_extra_tickers"])
    selected_tickers = _deduplicate(st.session_state["quant_selected_tickers"] + extra_tickers)
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
            button_type = "primary" if st.session_state["quant_horizon"] == label else "secondary"
            if column.button(label, key=f"quant-horizon-{label}", use_container_width=True, type=button_type):
                st.session_state["quant_horizon"] = label
    return st.session_state["quant_horizon"]


def _performance_card(title: str, ticker: str, expected_return: float) -> str:
    direction = "positive" if expected_return >= 0 else "negative"
    arrow = "&uarr;" if expected_return >= 0 else "&darr;"
    return (
        f"<div class='alfa-card'>"
        f"<div class='alfa-card-label'>{title}</div>"
        f"<div class='alfa-card-ticker'>{ticker}</div>"
        f"<div class='alfa-card-pill {direction}'>{arrow} {abs(expected_return) * 100:.0f}%</div>"
        f"</div>"
    )


CHART_COLORS_CLEAN = [
    "#2563EB", "#16A34A", "#4979f6", "#D97706", "#7C3AED",
    "#0891B2", "#DB2777", "#65A30D", "#EA580C",
]


def _apply_clean_chart_style(ax, fig) -> None:
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")
    ax.grid(True, axis="y", color="#F3F4F6", linewidth=0.9, linestyle="-")
    ax.grid(False, axis="x")
    ax.tick_params(axis="both", colors="#6B7280", labelsize=9)
    for spine in ax.spines.values():
        spine.set_visible(False)


def _plot_projection(history: pd.DataFrame, projected: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    _apply_clean_chart_style(ax, fig)

    for index, column in enumerate(history.columns):
        color = CHART_COLORS_CLEAN[index % len(CHART_COLORS_CLEAN)]
        ax.plot(history.index, history[column], linewidth=2, color=color, label=column)
        if column in projected.columns:
            ax.plot(projected.index, projected[column], linewidth=2, color=color, linestyle="--")

    if not history.empty:
        cutoff = history.index.max()
        ax.axvline(cutoff, color="#D1D5DB", linestyle=":", linewidth=1.2, alpha=0.9)

    ax.set_ylabel("Preço normalizado", color="#6B7280", fontsize=10)
    ax.set_xlabel("Data", color="#6B7280", fontsize=10)

    legend = ax.legend(
        title="Ticker",
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        frameon=False,
        labelcolor="#374151",
        borderaxespad=0,
        handlelength=0.8,
        handletextpad=0.5,
        fontsize=9,
    )
    if legend is not None:
        legend.get_title().set_color("#6B7280")
        legend.get_title().set_fontsize(9)

    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def render_quant_projections_page() -> None:
    _bootstrap_state()
    _render_styles()

    st.title("Projeções Quant")
    st.caption("Projeção simples de preços futuros sem data leakage, usando apenas histórico até a última data disponível.")

    option_pool, selected_tickers = _build_selection()
    left_col, right_col = st.columns([1.1, 3.2], gap="large")

    with left_col:
        with st.container(border=True):
            st.markdown("##### Stock tickers")
            st.multiselect(
                "Tickers",
                options=option_pool,
                key="quant_selected_tickers",
                label_visibility="collapsed",
                placeholder="Selecione os tickers",
            )
            st.text_input(
                "Adicionar tickers",
                key="quant_extra_tickers",
                placeholder="Ex.: PETR4, VALE3.SA, ASML",
            )

        horizon_label = _render_horizon_selector()

    if not selected_tickers:
        with right_col:
            st.info("Selecione pelo menos um ticker para exibir as projeções.")
        return

    if len(selected_tickers) > 9:
        selected_tickers = selected_tickers[:9]
        st.warning("A visualização foi limitada aos 9 primeiros tickers para manter a leitura do gráfico.")

    try:
        with st.spinner("Calculando projeções..."):
            result = fetch_ticker_quant_projection(tuple(selected_tickers), HORIZON_OPTIONS[horizon_label])
    except MarketDataError as exc:
        with right_col:
            st.error(str(exc))
        return
    except Exception:
        with right_col:
            st.error("Ocorreu um erro inesperado ao montar as projeções.")
        return

    if result.invalid_tickers:
        st.warning("Tickers ignorados por falta de histórico suficiente: " + ", ".join(result.invalid_tickers))

    if not result.expected_returns:
        with right_col:
            st.info("Não há dados suficientes para gerar projeções.")
        return

    sorted_returns = sorted(result.expected_returns.items(), key=lambda item: item[1], reverse=True)
    best_ticker, best_return = sorted_returns[0]
    worst_ticker, worst_return = sorted_returns[-1]

    with left_col:
        summary_col_1, summary_col_2 = st.columns(2)
        summary_col_1.markdown(_performance_card("Highest expected", best_ticker, best_return), unsafe_allow_html=True)
        summary_col_2.markdown(_performance_card("Lowest expected", worst_ticker, worst_return), unsafe_allow_html=True)

    with right_col:
        with st.container(border=True):
            history = result.historical_history.copy()
            projected = result.projected_history.copy()
            history.index = pd.to_datetime(history.index, errors="coerce")
            projected.index = pd.to_datetime(projected.index, errors="coerce")
            history = history[history.index.notna()]
            projected = projected[projected.index.notna()]
            _plot_projection(history, projected)
            if not history.empty and not projected.empty:
                start_date = history.index.min().strftime("%d/%m/%Y")
                cutoff_date = history.index.max().strftime("%d/%m/%Y")
                forecast_end = projected.index.max().strftime("%d/%m/%Y")
                st.caption(
                    f"Histórico: {start_date} a {cutoff_date}. Projeção simples: {cutoff_date} a {forecast_end}. Linha tracejada = previsão."
                )
                st.caption("Modelo: drift exponencial em retornos logarítmicos, estimado somente com dados históricos até a data de corte.")
