from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.config_repository import ConfigRepository
from core.portfolio_repository import PortfolioRepository
from services.black_litterman import BLView, run_black_litterman, save_views_to_csv
from services.markowitz import (
    MarkowitzFullResult,
    estimate_covariance,
    estimate_expected_returns,
    run_markowitz,
)
from services.portfolio_analytics import (
    _available_asset_columns,
    _normalized_weights,
)


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_VIEWS_CSV = _PROJECT_ROOT / "bl_views.csv"
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
    bl_vol: float | None = None,
    bl_ret: float | None = None,
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

    # BL portfolio
    if bl_vol is not None and bl_ret is not None:
        fig.add_trace(go.Scatter(
            x=[bl_vol], y=[bl_ret],
            mode="markers",
            marker=dict(size=12, symbol="hexagon", color="#059669", line=dict(color="#064E3B", width=1.5)),
            name="Carteira BL",
            hovertemplate="Vol: %{x:.2%}<br>Ret: %{y:.2%}<extra></extra>",
        ))

    # Collect all x/y data for axis ranges
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
# Render functions
# ------------------------------------------------------------------

def _render_weights_table(result: MarkowitzFullResult, bl_weights: pd.Series | None = None):
    """Weights comparison table."""
    cols_dict = {
        "Ativo": result.asset_names,
        "Max Sharpe (%)": result.max_sharpe.weights.values * 100,
        "Mín. Vol. (%)": result.min_variance.weights.values * 100,
    }
    if bl_weights is not None:
        cols_dict["Black-Litterman (%)"] = bl_weights.reindex(result.asset_names).fillna(0).values * 100

    weights_df = pd.DataFrame(cols_dict)
    weights_df["Ativo"] = weights_df["Ativo"].str.replace(".SA", "", regex=False)
    # Show only meaningful weights
    threshold_cols = [c for c in weights_df.columns if c != "Ativo"]
    mask = weights_df[threshold_cols].abs().max(axis=1) > 0.1
    weights_df = weights_df[mask].sort_values("Max Sharpe (%)", ascending=False)

    col_config = {c: st.column_config.NumberColumn(format="%.2f") for c in threshold_cols}
    st.dataframe(weights_df, use_container_width=True, hide_index=True, column_config=col_config)


def _render_summary_metrics(result: MarkowitzFullResult, bl_ret: float | None = None, bl_vol: float | None = None):
    cols = st.columns(3 if bl_ret is None else 4)
    cols[0].metric("Max Sharpe", f"{result.max_sharpe.expected_return:.1%} ret · {result.max_sharpe.volatility:.1%} vol")
    cols[1].metric("Mín. Volatilidade", f"{result.min_variance.expected_return:.1%} ret · {result.min_variance.volatility:.1%} vol")
    if result.tangency:
        cols[2].metric("Tangência", f"{result.tangency.expected_return:.1%} ret · {result.tangency.volatility:.1%} vol")
    if bl_ret is not None and bl_vol is not None:
        cols[-1].metric("Black-Litterman", f"{bl_ret:.1%} ret · {bl_vol:.1%} vol")


# ------------------------------------------------------------------
# Main page
# ------------------------------------------------------------------

def render_markowitz_page() -> None:
    portfolio_df, historical_df = _load_data()

    st.title("Fronteira Eficiente")
    st.caption("Simulação, otimização e alocação Black-Litterman.")

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
        ["Simulação", "Otimização", "Black-Litterman"],
        horizontal=True,
        label_visibility="collapsed",
    )

    # ── Premisses sidebar ────────────────────────────────────────
    config_repo = ConfigRepository(base_path=_PROJECT_ROOT)
    config = config_repo.load()

    with st.expander("⚙️  Parâmetros", expanded=False):
        p1, p2, p3, p4 = st.columns(4)
        rf_val = p1.number_input("Rf (a.a.)", value=config.risk_free_rate, min_value=0.0, max_value=1.0, step=0.0025, format="%.4f")
        emrp_val = p2.number_input("EMRP (a.a.)", value=config.emrp, min_value=0.0, max_value=0.30, step=0.005, format="%.4f")
        max_w = p3.number_input("Peso máx. por ativo", value=config.max_weight, min_value=0.05, max_value=1.0, step=0.05, format="%.2f")

        ret_source = "historical"
        if mode_tab == "Black-Litterman":
            ret_source = "bl_posterior"
        else:
            ret_source = p4.selectbox("Retornos", ["Históricos", "CAPM forward"], index=0)
            ret_source = "capm" if ret_source == "CAPM forward" else "historical"

        if rf_val != config.risk_free_rate or emrp_val != config.emrp or max_w != config.max_weight:
            config_repo.update(risk_free_rate=rf_val, emrp=emrp_val, max_weight=max_w)

    # ── BL Views (only in BL mode) ───────────────────────────────
    bl_result = None
    bl_weights = None
    bl_vol, bl_ret = None, None

    if mode_tab == "Black-Litterman":
        st.subheader("Views do gestor")
        with st.expander("⚙️  Parâmetros Black-Litterman", expanded=False):
            bl_c1, bl_c2, bl_c3 = st.columns(3)
            delta_val = bl_c1.number_input("δ (aversão ao risco)", value=config.delta, min_value=0.1, max_value=10.0, step=0.1, format="%.2f")
            tau_val = bl_c2.number_input("τ", value=config.tau, min_value=0.001, max_value=0.5, step=0.005, format="%.3f")
            use_current = bl_c3.checkbox("Usar pesos atuais como market weights", value=True)
            if delta_val != config.delta or tau_val != config.tau:
                config_repo.update(delta=delta_val, tau=tau_val)

        # Editable views
        views_key = "bl_views_editor"
        if views_key not in st.session_state:
            st.session_state[views_key] = pd.DataFrame({
                "Tipo": ["absolute"],
                "Ativo 1": [asset_names[0].replace(".SA", "") if asset_names else ""],
                "Ativo 2": [""],
                "Retorno esperado (a.a.)": [0.14],
                "Confiança (0-1)": [0.7],
                "Ativo": [True],
            })

        edited_views = st.data_editor(
            st.session_state[views_key],
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "Tipo": st.column_config.SelectboxColumn(options=["absolute", "relative"]),
                "Retorno esperado (a.a.)": st.column_config.NumberColumn(format="%.4f"),
                "Confiança (0-1)": st.column_config.NumberColumn(min_value=0.01, max_value=0.99, format="%.2f"),
            },
        )
        st.session_state[views_key] = edited_views

        # Parse views
        bl_views: list[BLView] = []
        for idx, row in edited_views.iterrows():
            if not row.get("Ativo", True):
                continue
            a1 = str(row.get("Ativo 1", "")).strip()
            if not a1:
                continue
            # Try to match with .SA suffix
            a1_full = a1 if a1 in asset_names else f"{a1}.SA" if f"{a1}.SA" in asset_names else a1
            a2_raw = str(row.get("Ativo 2", "")).strip()
            a2_full = None
            if a2_raw:
                a2_full = a2_raw if a2_raw in asset_names else f"{a2_raw}.SA" if f"{a2_raw}.SA" in asset_names else a2_raw

            bl_views.append(BLView(
                view_id=idx,
                view_type=str(row.get("Tipo", "absolute")),
                asset_1=a1_full,
                asset_2=a2_full,
                expected_excess_return=float(row.get("Retorno esperado (a.a.)", 0.0)),
                confidence=float(row.get("Confiança (0-1)", 0.5)),
                enabled=True,
            ))

        if bl_views:
            prices = historical_df[asset_names].ffill().bfill()
            cov_df = estimate_covariance(prices)
            mkt_w = _normalized_weights(portfolio_df, asset_names) if use_current else pd.Series(np.ones(len(asset_names)) / len(asset_names), index=asset_names)

            bl_result = run_black_litterman(
                cov=cov_df,
                market_weights=mkt_w,
                views=bl_views,
                asset_universe=asset_names,
                risk_aversion=delta_val,
                tau=tau_val,
                rf=rf_val,
                max_weight=max_w,
            )

            if bl_result is not None:
                bl_weights = bl_result.optimal_weights
                bl_ret = float(bl_weights.values @ bl_result.posterior_mu.reindex(asset_names).fillna(0).values)
                bl_vol = float(np.sqrt(bl_weights.values @ cov_df.reindex(index=asset_names, columns=asset_names).fillna(0).values @ bl_weights.values))

    # ── Run Markowitz ────────────────────────────────────────────
    run_sim = mode_tab == "Simulação"
    bl_posterior = bl_result.posterior_mu if bl_result is not None else None

    result = run_markowitz(
        prices=historical_df,
        asset_names=asset_names,
        rf=rf_val,
        return_method=ret_source if mode_tab != "Black-Litterman" else "historical",
        emrp=emrp_val,
        bl_posterior=bl_posterior,
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
    fig = _build_frontier_chart(
        result,
        show_cloud=run_sim,
        current_vol=cur_vol, current_ret=cur_ret,
        bl_vol=bl_vol, bl_ret=bl_ret,
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # Weights table
    st.subheader("Carteiras otimizadas")
    _render_weights_table(result, bl_weights)

    # Summary KPIs
    _render_summary_metrics(result, bl_ret, bl_vol)

    # BL comparison table
    if bl_result is not None:
        st.subheader("Retornos: Equilíbrio vs. Posterior")
        comp_df = pd.DataFrame({
            "Ativo": [a.replace(".SA", "") for a in asset_names],
            "Retorno de Equilíbrio": bl_result.prior_pi.values,
            "Retorno Posterior (BL)": bl_result.posterior_mu.values,
            "Diferença": bl_result.posterior_mu.values - bl_result.prior_pi.values,
        })
        st.dataframe(
            comp_df.style.format({
                "Retorno de Equilíbrio": "{:.2%}",
                "Retorno Posterior (BL)": "{:.2%}",
                "Diferença": "{:+.2%}",
            }),
            use_container_width=True,
            hide_index=True,
        )
