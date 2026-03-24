from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.config_repository import ConfigRepository
from core.portfolio_repository import PortfolioRepository
from services.markowitz import (
    MarkowitzFullResult,
    run_markowitz,
)
from services.portfolio_analytics import (
    _available_asset_columns,
    _normalized_weights,
)


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_CUSTOM_BLUES = [[0.0, "#93c5fd"], [0.35, "#4979f6"], [0.7, "#1e379b"], [1.0, "#102170"]]


# ------------------------------------------------------------------
# Data helpers
# ------------------------------------------------------------------

def _load_data():
    if "portfolio_df" in st.session_state and "historical_df" in st.session_state:
        return st.session_state["portfolio_df"], st.session_state["historical_df"]
    repo = PortfolioRepository(base_path=_PROJECT_ROOT)
    snapshot = repo.load_snapshot()
    return snapshot.portfolio, snapshot.historical


def _current_portfolio_point(portfolio_df, historical_df, asset_names, mu, cov):
    """Return (vol, ret) of the current portfolio weights."""
    weights = _normalized_weights(portfolio_df, asset_names)
    if weights.empty:
        return None, None
    w = weights.reindex(asset_names).fillna(0.0).values
    ret = float(w @ mu.reindex(asset_names).fillna(0.0).values)
    vol = float(np.sqrt(w @ cov.reindex(index=asset_names, columns=asset_names).fillna(0.0).values @ w))
    return vol, ret


# ------------------------------------------------------------------
# Chart builder
# ------------------------------------------------------------------

def _build_frontier_chart(
    result: MarkowitzFullResult,
    show_cloud: bool = True,
    current_vol: float | None = None,
    current_ret: float | None = None,
) -> go.Figure:
    fig = go.Figure()

    # Monte Carlo cloud
    if show_cloud and not result.cloud.empty:
        fig.add_trace(go.Scatter(
            x=result.cloud["volatilidade"], y=result.cloud["retorno"],
            mode="markers",
            marker=dict(size=3.5, color=result.cloud["sharpe"], colorscale=_CUSTOM_BLUES,
                        showscale=True, colorbar=dict(title="Sharpe", title_side="right", thickness=12, len=0.6),
                        opacity=0.40),
            name="Portfólios simulados",
            hovertemplate="Vol: %{x:.2%}<br>Ret: %{y:.2%}<br>Sharpe: %{marker.color:.2f}<extra></extra>",
        ))

    # Efficient frontier
    if not result.frontier.empty:
        mv_ret = result.min_variance.expected_return
        upper = result.frontier[result.frontier["retorno"] >= mv_ret - 1e-6]
        lower = result.frontier[result.frontier["retorno"] <= mv_ret + 1e-6]
        if not upper.empty:
            fig.add_trace(go.Scatter(
                x=upper["volatilidade"], y=upper["retorno"],
                mode="lines", line=dict(color="#1e379b", width=2.5),
                name="Fronteira eficiente", hoverinfo="skip",
            ))
        if not lower.empty:
            fig.add_trace(go.Scatter(
                x=lower["volatilidade"], y=lower["retorno"],
                mode="lines", line=dict(color="#93c5fd", width=1.5, dash="dash"),
                name="Fronteira ineficiente", hoverinfo="skip",
            ))

    # CML (Capital Market Line) from Rf through tangency
    if result.tangency and result.tangency.volatility > 1e-6:
        tg = result.tangency
        cml_slope = (tg.expected_return - result.rf) / tg.volatility
        cml_max_vol = tg.volatility * 1.8
        cml_x = [0, cml_max_vol]
        cml_y = [result.rf, result.rf + cml_slope * cml_max_vol]
        fig.add_trace(go.Scatter(
            x=cml_x, y=cml_y,
            mode="lines", line=dict(color="#059669", width=1.5, dash="dot"),
            name="CML", hoverinfo="skip",
        ))

    # Max Sharpe
    fig.add_trace(go.Scatter(
        x=[result.max_sharpe.volatility], y=[result.max_sharpe.expected_return],
        mode="markers",
        marker=dict(size=14, symbol="star", color="#4979f6", line=dict(color="#1e379b", width=1.5)),
        name="Máximo Sharpe",
        hovertemplate="Vol: %{x:.2%}<br>Ret: %{y:.2%}<extra></extra>",
    ))

    # Min Variance
    fig.add_trace(go.Scatter(
        x=[result.min_variance.volatility], y=[result.min_variance.expected_return],
        mode="markers",
        marker=dict(size=10, symbol="diamond", color="#1e379b", line=dict(color="#102170", width=1.5)),
        name="Mínima volatilidade",
        hovertemplate="Vol: %{x:.2%}<br>Ret: %{y:.2%}<extra></extra>",
    ))

    # Current portfolio
    if current_vol is not None and current_ret is not None:
        fig.add_trace(go.Scatter(
            x=[current_vol], y=[current_ret],
            mode="markers",
            marker=dict(size=12, symbol="circle", color="#EF4444", line=dict(color="#991B1B", width=1.5)),
            name="Carteira atual",
            hovertemplate="Vol: %{x:.2%}<br>Ret: %{y:.2%}<extra></extra>",
        ))

    # Axis ranges
    all_x, all_y = [], []
    for trace in fig.data:
        all_x.extend([v for v in trace.x if v is not None and np.isfinite(v)])
        all_y.extend([v for v in trace.y if v is not None and np.isfinite(v)])

    if all_x and all_y:
        pad_x = (max(all_x) - min(all_x)) * 0.06
        pad_y = (max(all_y) - min(all_y)) * 0.06
        x_range = [min(all_x) - pad_x, max(all_x) + pad_x]
        y_range = [min(all_y) - pad_y, max(all_y) + pad_y]
    else:
        x_range, y_range = None, None

    fig.update_layout(
        title=dict(text="Fronteira Eficiente de Markowitz", font=dict(color="#111827", size=14, family="Inter"), pad=dict(b=10)),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#6B7280", size=11, family="Inter"),
        margin=dict(l=0, r=20, t=60, b=40), hovermode="closest",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, title=""),
    )
    fig.update_xaxes(
        title_text="Volatilidade anualizada", showgrid=True, gridcolor="#E5E7EB", gridwidth=1,
        zeroline=False, showline=True, linecolor="#E5E7EB", linewidth=1, tickformat=".0%",
        range=x_range,
    )
    fig.update_yaxes(
        title_text="Retorno anualizado esperado", showgrid=True, gridcolor="#E5E7EB", gridwidth=1,
        zeroline=False, showline=False, tickformat=".0%",
        range=y_range,
    )

    return fig


# ------------------------------------------------------------------
# Main page
# ------------------------------------------------------------------

def render_markowitz_page() -> None:
    portfolio_df, historical_df = _load_data()

    st.title("Fronteira Eficiente")
    st.caption("Simulação e otimização de carteiras pela teoria de Markowitz.")

    if portfolio_df.empty or historical_df.empty:
        st.info("Monte o portfolio e atualize o histórico na página principal para liberar a análise de Markowitz.")
        return

    asset_names = _available_asset_columns(portfolio_df, historical_df)
    if len(asset_names) < 2:
        st.info("São necessários pelo menos dois ativos válidos para a análise.")
        return

    # ── Mode selection ───────────────────────────────────────────
    mode_tab = st.radio(
        "Modo de análise",
        ["Simulação", "Otimização"],
        horizontal=True,
        label_visibility="collapsed",
    )

    # ── Parameters ───────────────────────────────────────────────
    config_repo = ConfigRepository(base_path=_PROJECT_ROOT)
    config = config_repo.load()

    with st.expander("⚙️  Parâmetros", expanded=False):
        p1, p2, p3, p4 = st.columns(4)
        rf_val = p1.number_input("Rf (a.a.)", value=config.risk_free_rate, min_value=0.0, max_value=1.0, step=0.0025, format="%.4f")
        emrp_val = p2.number_input("EMRP (a.a.)", value=config.emrp, min_value=0.0, max_value=0.30, step=0.005, format="%.4f")
        max_w = p3.number_input("Peso máx. por ativo", value=config.max_weight, min_value=0.05, max_value=1.0, step=0.05, format="%.2f")
        ret_source = p4.selectbox("Retornos", ["Históricos", "CAPM forward"], index=0)
        ret_source = "capm" if ret_source == "CAPM forward" else "historical"

        if rf_val != config.risk_free_rate or emrp_val != config.emrp or max_w != config.max_weight:
            config_repo.update(risk_free_rate=rf_val, emrp=emrp_val, max_weight=max_w)

    # ── Run Markowitz ────────────────────────────────────────────
    run_sim = mode_tab == "Simulação"

    result = run_markowitz(
        prices=historical_df,
        asset_names=asset_names,
        rf=rf_val,
        return_method=ret_source,
        emrp=emrp_val,
        max_weight=max_w,
        run_simulation=run_sim,
        n_frontier_points=60,
    )

    if result is None:
        st.info("São necessários pelo menos dois ativos válidos e cerca de 21 observações.")
        return

    # Current portfolio point
    cur_vol, cur_ret = _current_portfolio_point(portfolio_df, historical_df, asset_names, result.mu, result.cov)

    # Chart
    fig = _build_frontier_chart(result, show_cloud=run_sim, current_vol=cur_vol, current_ret=cur_ret)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # Weights table
    st.subheader("Carteiras otimizadas")
    weights_df = pd.DataFrame({
        "Ativo": result.asset_names,
        "Max Sharpe (%)": result.max_sharpe.weights.values * 100,
        "Mín. Vol. (%)": result.min_variance.weights.values * 100,
    })
    weights_df["Ativo"] = weights_df["Ativo"].str.replace(".SA", "", regex=False)
    mask = weights_df[["Max Sharpe (%)", "Mín. Vol. (%)"]].abs().max(axis=1) > 0.1
    weights_df = weights_df[mask].sort_values("Max Sharpe (%)", ascending=False)

    st.dataframe(
        weights_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Max Sharpe (%)": st.column_config.NumberColumn(format="%.2f"),
            "Mín. Vol. (%)": st.column_config.NumberColumn(format="%.2f"),
        },
    )

    # Summary KPIs
    cols = st.columns(3)
    cols[0].metric("Max Sharpe", f"{result.max_sharpe.expected_return:.1%} ret · {result.max_sharpe.volatility:.1%} vol")
    cols[1].metric("Mín. Volatilidade", f"{result.min_variance.expected_return:.1%} ret · {result.min_variance.volatility:.1%} vol")
    if result.tangency:
        cols[2].metric("Tangência", f"{result.tangency.expected_return:.1%} ret · {result.tangency.volatility:.1%} vol")
