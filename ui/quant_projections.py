from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.portfolio_repository import PortfolioRepository
from services.market_data import MarketDataError, fetch_ticker_quant_projection


DEFAULT_TICKERS = ["AAPL", "MSFT", "GOOGL", "NVDA", "AMZN", "TSLA", "META"]
HORIZON_OPTIONS = {
    "1 Month": "1mo",
    "3 Months": "3mo",
    "6 Months": "6mo",
    "1 Year": "1y",
    "2 Years": "2y",
    "5 Years": "5y",
}
MODEL_OPTIONS = ["Drift Exponencial", "Holt-Winters", "ARIMA"]
CHART_COLORS_CLEAN = [
    "#4979f6", "#1e379b", "#102170"
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
    if "quant_model" not in st.session_state:
        st.session_state["quant_model"] = "Drift Exponencial"


def _render_styles() -> None:
    st.markdown(
        """
        <style>
        .alfa-card {
            background: #FFFFFF;
            border: 1px solid #E5E7EB;
            border-radius: 10px;
            padding: 1.1rem 1.2rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.06);
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
            display: inline-flex;
            align-items: center;
            border-radius: 6px;
            padding: 0.25rem 0.65rem;
            font-size: 0.88rem;
            font-weight: 600;
        }
        .alfa-card-pill.positive {
            background: #ECFDF5;
            color: #059669;
        }
        .alfa-card-pill.negative {
            background: #FEF2F2;
            color: #EF4444;
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
    st.write("**Horizonte de projeção**")
    horizon_rows = [
        ["1 Month", "3 Months", "6 Months"],
        ["1 Year", "2 Years", "5 Years"],
    ]
    for row in horizon_rows:
        columns = st.columns(len(row))
        for column, label in zip(columns, row):
            display_label = label.replace("Month", "Mês").replace("Months", "Meses").replace("Year", "Ano").replace("Years", "Anos")
            button_type = "primary" if st.session_state["quant_horizon"] == label else "secondary"
            if column.button(display_label, key=f"quant-horizon-{label}", use_container_width=True, type=button_type):
                st.session_state["quant_horizon"] = label
    return st.session_state["quant_horizon"]


def _performance_card(title: str, ticker: str, expected_return: float) -> str:
    direction = "positive" if expected_return >= 0 else "negative"
    arrow = "↑" if expected_return >= 0 else "↓"
    return (
        f"<div class='alfa-card'>"
        f"<div class='alfa-card-label'>{title}</div>"
        f"<div class='alfa-card-ticker'>{ticker}</div>"
        f"<div class='alfa-card-pill {direction}'>{arrow} {abs(expected_return) * 100:.2f}%</div>"
        f"</div>"
    )


def _apply_alfa_style(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#6B7280", size=11, family="Inter"),
        margin=dict(l=0, r=20, t=10, b=40),
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


def _plot_projection(history: pd.DataFrame, projected: pd.DataFrame) -> None:
    fig = go.Figure()

    for index, column in enumerate(history.columns):
        color = CHART_COLORS_CLEAN[index % len(CHART_COLORS_CLEAN)]
        # Historical line
        fig.add_trace(go.Scatter(
            x=history.index,
            y=history[column],
            mode="lines",
            line=dict(color=color, width=2),
            name=f"{column} (Histórico)"
        ))
        # Projected line
        if column in projected.columns:
            fig.add_trace(go.Scatter(
                x=projected.index,
                y=projected[column],
                mode="lines",
                line=dict(color=color, width=2, dash="dash"),
                name=f"{column} (Projeção)"
            ))

    if not history.empty:
        cutoff = history.index.max()
        fig.add_vline(x=cutoff, line_width=1.5, line_dash="dash", line_color="#9CA3AF")

    _apply_alfa_style(fig)
    fig.update_yaxes(title_text="Base 1.0 (Normalizado)")
    fig.update_xaxes(title_text="")

    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def render_quant_projections_page() -> None:
    _bootstrap_state()
    _render_styles()

    st.title("Projeções Quantitativas")
    st.caption("Projeção de preços futuros usando modelos de séries temporais aplicados ao histórico de ativos.")

    option_pool, selected_tickers = _build_selection()

    st.markdown('<div class="alfa-section-title">Controles do Modelo</div>', unsafe_allow_html=True)
    control_col_1, control_col_2, control_col_3 = st.columns([1.1, 1, 1], gap="large")

    with control_col_1:
        st.write("**Seleção de ativos**")
        st.multiselect(
            "Tickers principais",
            options=option_pool,
            key="quant_selected_tickers",
            label_visibility="collapsed",
            placeholder="Ex.: AAPL, MSFT, PETR4.SA",
        )
        st.text_input(
            "Tickers extras (separados por vírgula)",
            key="quant_extra_tickers",
            placeholder="Ex.: VIVT3.SA, ASML",
        )

    with control_col_2:
        horizon_label = _render_horizon_selector()

    with control_col_3:
        st.write("**Modelo preditivo**")
        model_name = st.selectbox(
            "Modelo",
            options=MODEL_OPTIONS,
            key="quant_model",
            label_visibility="collapsed",
        )
        st.caption(f"Tratamento empregado: {model_name}")

    st.markdown("---")

    if not selected_tickers:
        st.info("Selecione pelo menos um ticker para exibir as projeções.")
        return

    if len(selected_tickers) > 9:
        selected_tickers = selected_tickers[:9]
        st.warning("A visualização foi limitada aos 9 primeiros tickers para manter a leitura do gráfico.")

    try:
        with st.spinner(f"Calculando projeções via {model_name}..."):
            historical_df = st.session_state.get("historical_df")
            result = fetch_ticker_quant_projection(tuple(selected_tickers), HORIZON_OPTIONS[horizon_label], model_name, historical_df=historical_df)
    except MarketDataError as exc:
        st.error(str(exc))
        return
    except Exception:
        st.error("Ocorreu um erro inesperado ao montar as projeções.")
        return

    if result.invalid_tickers:
        st.warning("Tickers ignorados por falta de histórico suficiente: " + ", ".join(result.invalid_tickers))

    if not result.expected_returns:
        st.info("Não há dados suficientes para gerar projeções.")
        return

    sorted_returns = sorted(result.expected_returns.items(), key=lambda item: item[1], reverse=True)
    best_ticker, best_return = sorted_returns[0]
    worst_ticker, worst_return = sorted_returns[-1]

    st.markdown('<div class="alfa-section-title">Resultados das Projeções</div>', unsafe_allow_html=True)

    summary_col_1, summary_col_2, empty_col = st.columns([1, 1, 2])
    with summary_col_1:
        st.markdown(_performance_card("Maior alta esperada", best_ticker, best_return), unsafe_allow_html=True)
    with summary_col_2:
        st.markdown(_performance_card("Maior queda esperada", worst_ticker, worst_return), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

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
                f"Histórico (*linha sólida*): {start_date} a {cutoff_date}. "
                f"Projeção (*linha tracejada*): até {forecast_end}."
            )
            model_descriptions = {
                "Drift Exponencial": "Modelo de drift exponencial com base em retornos recentes e longo prazo.",
                "Holt-Winters": "Suavização exponencial simples (Holt) com tendência aditiva, sem sazonalidade.",
                "ARIMA": "Modelo Autoregressivo Integrado de Médias Móveis (1, 1, 0) para séries não estacionárias."
            }
            st.caption(f"Metodologia: {model_descriptions.get(model_name, '')}")
