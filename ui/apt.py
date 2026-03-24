"""APT / Fatores page.

Select assets, configure factors and window, and view multifactor regression
results with significance, VIF, and factor contribution visuals.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.portfolio_repository import PortfolioRepository
from services.apt import run_apt_analysis
from services.portfolio_analytics import _active_assets

_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_data():
    if "portfolio_df" in st.session_state and "historical_df" in st.session_state:
        return st.session_state["portfolio_df"], st.session_state["historical_df"]
    repo = PortfolioRepository(base_path=_PROJECT_ROOT)
    snapshot = repo.load_snapshot()
    return snapshot.portfolio, snapshot.historical


def render_apt_page() -> None:
    portfolio_df, historical_df = _load_data()

    st.title("APT — Análise Multifatorial")
    st.caption("Regressão de retornos contra fatores macroeconômicos (IBOV, IPCA, PTAX, Selic).")

    if portfolio_df.empty or historical_df.empty:
        st.info("Monte o portfolio e atualize o histórico na página principal para acessar a análise APT.")
        return

    active = _active_assets(portfolio_df)
    if active.empty:
        st.info("Nenhum ativo ativo no portfolio.")
        return

    tickers = [t for t in active["ticker"].tolist() if t in historical_df.columns]
    if not tickers:
        st.info("Nenhum ativo com histórico disponível para a análise APT.")
        return

    # ── Controls ─────────────────────────────────────────────────
    with st.expander("Parâmetros APT", expanded=False):
        c1, c2, c3 = st.columns(3)
        window_years = c1.selectbox("Janela (anos)", [1, 2, 3, 5, 10], index=2)
        lags_val = c2.number_input("Defasagens (meses)", min_value=0, max_value=6, value=0)

        all_factors = ["Mercado (IBOV)", "Inflação (IPCA)", "Câmbio (PTAX)", "ΔSelic"]
        enabled_factors = c3.multiselect("Fatores", all_factors, default=all_factors)

    end_date = datetime.now()
    start_date = end_date - timedelta(days=int(window_years * 365))

    # ── Run analysis ─────────────────────────────────────────────
    with st.spinner("Buscando fatores e executando regressões..."):
        result = run_apt_analysis(
            asset_prices=historical_df,
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            lags=lags_val,
            enabled_factors=enabled_factors if enabled_factors else None,
        )

    if result is None:
        st.warning(
            "Não foi possível executar a análise APT. Possíveis causas:\n"
            "- Pacote `python-bcb` não instalado\n"
            "- Séries macro indisponíveis no período\n"
            "- Dados insuficientes para regressão (mínimo ~12 meses)"
        )
        return

    # ── Betas table ──────────────────────────────────────────────
    st.subheader("Betas de Fatores (APT)")

    rows = []
    for reg in result.regressions:
        row = {
            "Ativo": reg.ticker.replace(".SA", ""),
            "Intercepto": reg.intercept,
        }
        for factor in result.factor_names:
            row[f"β {factor}"] = reg.betas.get(factor, 0)
            row[f"p {factor}"] = reg.p_values.get(factor, 1)
        row["R² adj."] = reg.r_squared_adj
        row["F-stat"] = reg.f_stat
        row["p(F)"] = reg.f_pvalue
        rows.append(row)

    betas_df = pd.DataFrame(rows)

    # Format
    fmt = {"Intercepto": "{:.4f}", "R² adj.": "{:.4f}", "F-stat": "{:.2f}", "p(F)": "{:.4f}"}
    for f in result.factor_names:
        fmt[f"β {f}"] = "{:.4f}"
        fmt[f"p {f}"] = "{:.4f}"

    st.dataframe(
        betas_df.style.format(fmt).background_gradient(
            subset=["R² adj."], cmap="Blues", vmin=0, vmax=1
        ),
        use_container_width=True,
        hide_index=True,
    )

    # ── Significance heatmap ─────────────────────────────────────
    st.subheader("Significância dos fatores (p-values)")
    p_cols = [f"p {f}" for f in result.factor_names if f"p {f}" in betas_df.columns]
    if p_cols:
        pval_df = betas_df[["Ativo"] + p_cols].set_index("Ativo")
        pval_df.columns = [c.replace("p ", "") for c in pval_df.columns]
        st.dataframe(
            pval_df.style.format("{:.4f}").background_gradient(cmap="RdYlGn_r", vmin=0, vmax=0.15),
            use_container_width=True,
        )

    # ── VIF ──────────────────────────────────────────────────────
    if result.vif is not None and not result.vif.empty:
        st.subheader("VIF — Multicolinearidade")
        st.dataframe(
            result.vif.style.format({"VIF": "{:.2f}"}).background_gradient(subset=["VIF"], cmap="Oranges", vmin=1, vmax=10),
            use_container_width=True,
            hide_index=True,
        )
        st.caption("VIF > 5 indica multicolinearidade moderada; VIF > 10 indica multicolinearidade alta.")

    # ── Factor contribution chart ────────────────────────────────
    st.subheader("Contribuição dos fatores ao retorno")
    if result.regressions:
        # Use simple product: beta × factor mean return
        contrib_rows = []
        for reg in result.regressions:
            for factor, beta in reg.betas.items():
                contrib_rows.append({
                    "Ativo": reg.ticker.replace(".SA", ""),
                    "Fator": factor,
                    "Beta": beta,
                })
        contrib_df = pd.DataFrame(contrib_rows)

        if not contrib_df.empty:
            fig = go.Figure()
            for factor in result.factor_names:
                subset = contrib_df[contrib_df["Fator"] == factor]
                if subset.empty:
                    continue
                fig.add_trace(go.Bar(
                    x=subset["Ativo"],
                    y=subset["Beta"],
                    name=factor,
                ))
            fig.update_layout(
                barmode="group",
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#6B7280", size=11, family="Inter"),
                margin=dict(l=0, r=20, t=30, b=30),
                yaxis_title="Beta",
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            fig.update_xaxes(showgrid=False)
            fig.update_yaxes(showgrid=True, gridcolor="#E5E7EB")
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # ── Export ────────────────────────────────────────────────────
    st.subheader("Exportar")
    col_exp1, col_exp2 = st.columns(2)
    csv_data = betas_df.to_csv(index=False)
    col_exp1.download_button("📥 Exportar tabela (CSV)", csv_data, "apt_betas.csv", "text/csv")
