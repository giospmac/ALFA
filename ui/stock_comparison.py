from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.portfolio_repository import PortfolioRepository
from services.market_data import MarketDataError, fetch_ticker_comparison, compute_comparison_metrics


DEFAULT_TICKERS = [
    "ITUB4.SA", "BBAS3.SA", "BPAC11.SA", "RDOR3.SA", "WEGE3.SA", "ENGI11.SA",
    "PRIO3.SA", "SUZB3.SA", "MRFG3.SA", "MULT3.SA", "TEND3.SA", "RENT3.SA",
    "CURY3.SA", "RAIL3.SA", "MSFT", "NVDA", "GOOGL",
]
HORIZON_OPTIONS = {
    "1 Mês": "1mo",
    "3 Meses": "3mo",
    "6 Meses": "6mo",
    "1 Ano": "1y",
    "5 Anos": "5y",
    "10 Anos": "10y",
    "20 Anos": "20y",
}
CHART_COLORS = ["#4979f6", "#1e379b", "#102170", "#6fa0ff", "#3355cc", "#0d1a4a", "#8ab4ff", "#2a4da6", "#15306b"]


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
        st.session_state["comparison_horizon"] = "6 Meses"


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
    extra_tickers = _parse_extra_tickers(st.session_state["comparison_extra_tickers"])
    selected_tickers = _deduplicate(st.session_state["comparison_selected_tickers"] + extra_tickers)
    return option_pool, selected_tickers


def _performance_card(title: str, ticker: str, total_return: float) -> str:
    direction = "positive" if total_return >= 0 else "negative"
    arrow = "↑" if total_return >= 0 else "↓"
    return (
        f"<div class='alfa-card'>"
        f"<div class='alfa-card-label'>{title}</div>"
        f"<div class='alfa-card-ticker'>{ticker}</div>"
        f"<div class='alfa-card-pill {direction}'>{arrow} {abs(total_return) * 100:.2f}%</div>"
        f"</div>"
    )


def _apply_alfa_style(fig: go.Figure, title: str = "") -> go.Figure:
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(color="#111827", size=14, family="Inter"),
            pad=dict(b=10)
        ) if title else None,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#6B7280", size=11, family="Inter"),
        margin=dict(l=0, r=20, t=40 if title else 10, b=40),
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


# ── Charts ──────────────────────────────────────────────────────────────────


def _plot_cumulative_returns(history: pd.DataFrame) -> None:
    """Cumulative return (%) chart starting at 0%."""
    pct_history = (history / history.iloc[0] - 1) * 100
    fig = go.Figure()
    for idx, col in enumerate(pct_history.columns):
        fig.add_trace(go.Scatter(
            x=pct_history.index, y=pct_history[col],
            mode="lines",
            line=dict(color=CHART_COLORS[idx % len(CHART_COLORS)], width=2),
            name=col
        ))
    _apply_alfa_style(fig)
    fig.update_yaxes(ticksuffix="%")
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def _plot_drawdown(history: pd.DataFrame) -> None:
    """Comparative drawdown chart."""
    daily_returns = history.pct_change().dropna(how="all")
    fig = go.Figure()
    for idx, col in enumerate(daily_returns.columns):
        r = daily_returns[col].dropna()
        cum = (1 + r).cumprod()
        dd = (cum - cum.cummax()) / cum.cummax() * 100
        fig.add_trace(go.Scatter(
            x=dd.index, y=dd.values,
            mode="lines",
            line=dict(color=CHART_COLORS[idx % len(CHART_COLORS)], width=1.5),
            name=col
        ))
    _apply_alfa_style(fig)
    fig.update_yaxes(ticksuffix="%")
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def _plot_return_vs_volatility(metrics_list) -> None:
    """Grouped bar chart: return vs volatility per ticker."""
    tickers = [m.ticker for m in metrics_list]
    returns = [m.total_return * 100 for m in metrics_list]
    vols = [m.annualized_volatility * 100 for m in metrics_list]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=tickers, y=returns, name="Retorno Total (%)",
        marker_color="#4979f6", width=0.35
    ))
    fig.add_trace(go.Bar(
        x=tickers, y=vols, name="Volatilidade Anual. (%)",
        marker_color="#102170", width=0.35
    ))
    fig.update_layout(barmode="group")
    _apply_alfa_style(fig)
    fig.update_yaxes(ticksuffix="%")
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def _plot_correlation_heatmap(history: pd.DataFrame) -> None:
    """Correlation heatmap of daily returns."""
    corr = history.pct_change().dropna(how="all").corr()
    tickers = list(corr.columns)

    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=tickers,
        y=tickers,
        colorscale=[[0, "#97bdff"], [0.5, "#4979f6"], [1.0, "#102170"]],
        zmin=-1, zmax=1,
        text=[[f"{v:.2f}" for v in row] for row in corr.values],
        texttemplate="%{text}",
        textfont=dict(size=11, color="#FFFFFF"),
        hovertemplate="<b>%{x}</b> vs <b>%{y}</b><br>Correlação: %{z:.2f}<extra></extra>",
        colorbar=dict(
            title="Corr",
            thickness=12,
            outlinewidth=0,
        )
    ))
    _apply_alfa_style(fig)
    fig.update_layout(
        margin=dict(l=0, r=20, t=10, b=0),
        yaxis=dict(autorange="reversed"),
        xaxis=dict(side="bottom"),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def _render_metrics_table(metrics_list) -> None:
    """Indicator table with one row per ticker."""
    rows = []
    for m in metrics_list:
        rows.append({
            "Ticker": m.ticker,
            "Retorno Total (%)": round(m.total_return * 100, 2),
            "Vol. Anual. (%)": round(m.annualized_volatility * 100, 2),
            "Sharpe": round(m.sharpe_ratio, 2) if m.sharpe_ratio is not None else None,
            "Sortino": round(m.sortino_ratio, 2) if m.sortino_ratio is not None else None,
            "Max Drawdown (%)": round(m.max_drawdown * 100, 2),
            "Beta": round(m.beta, 2) if m.beta is not None else None,
            "Ret. Diário Médio (%)": round(m.avg_daily_return * 100, 4),
        })
    df = pd.DataFrame(rows)
    st.dataframe(
        df.style.set_properties(**{"text-align": "center"}),
        use_container_width=True,
        hide_index=True,
    )


# ── Main page ───────────────────────────────────────────────────────────────


def render_stock_comparison_page() -> None:
    _bootstrap_state()
    _render_styles()

    st.title("Comparador de Ações")
    st.caption("Compare a performance relativa de múltiplos tickers no período escolhido.")

    option_pool, selected_tickers = _build_selection()

    st.markdown('<div class="alfa-section-title">Controles</div>', unsafe_allow_html=True)
    control_col_1, control_col_2 = st.columns([1.1, 1], gap="large")

    with control_col_1:
        st.write("**Seleção de ativos**")
        st.multiselect(
            "Tickers principais",
            options=option_pool,
            key="comparison_selected_tickers",
            label_visibility="collapsed",
            placeholder="Ex.: AAPL, MSFT, PETR4.SA",
        )
        st.text_input(
            "Tickers extras (separados por vírgula)",
            key="comparison_extra_tickers",
            placeholder="Ex.: VIVT3.SA, ASML",
        )

    with control_col_2:
        st.write("**Horizonte de tempo**")
        horizon_label = st.pills(
            "Horizonte",
            options=list(HORIZON_OPTIONS.keys()),
            default=st.session_state["comparison_horizon"],
            key="comparison_horizon_pills",
            label_visibility="collapsed",
        )
        if horizon_label:
            st.session_state["comparison_horizon"] = horizon_label
        else:
            horizon_label = st.session_state["comparison_horizon"]

    st.markdown("---")

    if not selected_tickers:
        st.info("Selecione pelo menos um ticker para exibir a comparação.")
        return

    if len(selected_tickers) > 9:
        selected_tickers = selected_tickers[:9]
        st.warning("A visualização foi limitada aos 9 primeiros tickers para manter a leitura do gráfico focada.")

    try:
        with st.spinner("Consultando séries históricas..."):
            historical_df = st.session_state.get("historical_df")
            result = fetch_ticker_comparison(tuple(selected_tickers), HORIZON_OPTIONS[horizon_label], historical_df=historical_df)
    except MarketDataError as exc:
        st.error(str(exc))
        return
    except Exception:
        st.error("Ocorreu um erro inesperado ao montar a comparação.")
        return

    if result.invalid_tickers:
        st.warning("Tickers ignorados por falta de histórico suficiente na janela: " + ", ".join(result.invalid_tickers))

    if not result.total_returns:
        st.info("Não há dados suficientes para comparar os tickers selecionados neste período.")
        return

    # ── Performance summary ──────────────────────────────────────────────
    sorted_returns = sorted(result.total_returns.items(), key=lambda item: item[1], reverse=True)
    best_ticker, best_return = sorted_returns[0]
    worst_ticker, worst_return = sorted_returns[-1]

    st.markdown('<div class="alfa-section-title">Resultados da Comparação</div>', unsafe_allow_html=True)

    summary_col_1, summary_col_2, empty_col = st.columns([1, 1, 2])
    with summary_col_1:
        st.markdown(_performance_card("Melhor performance", best_ticker, best_return), unsafe_allow_html=True)
    with summary_col_2:
        st.markdown(_performance_card("Pior performance", worst_ticker, worst_return), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Charts ───────────────────────────────────────────────────────────
    history = result.close_history.copy() if result.close_history is not None else result.normalized_history.copy()
    history.index = pd.to_datetime(history.index, errors="coerce")
    history = history[history.index.notna()]

    if history.empty:
        st.info("Sem dados suficientes para gerar as visualizações.")
        return

    start_date = history.index.min().strftime("%d/%m/%Y")
    end_date = history.index.max().strftime("%d/%m/%Y")

    # 1) Cumulative returns
    st.subheader("Retorno Acumulado")
    _plot_cumulative_returns(history)
    st.caption(f"Janela analisada: {start_date} a {end_date}.")

    # 2) Drawdown
    st.subheader("Drawdown Comparativo")
    _plot_drawdown(history)

    # 3) Compute metrics
    with st.spinner("Calculando indicadores de mercado..."):
        metrics_list = compute_comparison_metrics(history)

    if metrics_list:
        # 4) Return vs Volatility
        st.subheader("Retorno vs Volatilidade")
        _plot_return_vs_volatility(metrics_list)

        # 5) Indicators table
        st.subheader("Indicadores Comparativos")
        _render_metrics_table(metrics_list)

    # 6) Correlation heatmap
    if len(history.columns) > 1:
        st.subheader("Correlação entre Ativos")
        _plot_correlation_heatmap(history)
