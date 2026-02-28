from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from core.portfolio_repository import PortfolioRepository
from services.portfolio_analytics import (
    accumulated_returns_table,
    capm_alpha_beta_correlation,
    correlation_matrix,
    individual_metrics,
    performance_indices,
    var_cvar_metrics,
)


MONTH_LABELS = ["Jan", "Fev", "Mar", "Abr", "Mai", "Jun", "Jul", "Ago", "Set", "Out", "Nov", "Dez"]


def _load_data():
    if "portfolio_df" in st.session_state and "historical_df" in st.session_state:
        total_pl = float(st.session_state.get("portfolio_total_pl", 100000000.0))
        return st.session_state["portfolio_df"], st.session_state["historical_df"], total_pl

    repo = PortfolioRepository(base_path=Path(__file__).resolve().parents[1])
    snapshot = repo.load_snapshot()
    invested_value = float(pd.to_numeric(snapshot.portfolio.get("valor_real"), errors="coerce").fillna(0).sum()) if not snapshot.portfolio.empty else 0.0
    total_pl = invested_value if invested_value > 0 else 100000000.0
    return snapshot.portfolio, snapshot.historical, total_pl


def _returns_table_to_frame(table_data: dict[int, dict[int, float]], yearly_table: dict[int, float]) -> pd.DataFrame:
    years = sorted(table_data.keys())[-5:]
    rows = []
    for year in years:
        row = {"Ano": year}
        for month_number, month_label in enumerate(MONTH_LABELS, start=1):
            value = table_data.get(year, {}).get(month_number)
            row[month_label] = value * 100 if value is not None else None
        annual_value = yearly_table.get(year)
        row["Acumulado (Ano)"] = annual_value * 100 if annual_value is not None else None
        rows.append(row)
    return pd.DataFrame(rows)


def render_risk_analysis_page() -> None:
    portfolio_df, historical_df, total_pl = _load_data()

    st.title("Indicadores de Risco")
    st.caption("Tabelas de rentabilidade, VaR, CVaR, CAPM, performance ajustada ao benchmark e correlacao.")

    if portfolio_df.empty or historical_df.empty:
        st.info("Monte o portfolio e atualize o historico na pagina principal antes de abrir esta analise.")
        return

    control_col_1, control_col_2 = st.columns([1, 1])
    unit = control_col_1.selectbox("Unidade do periodo", ["Ano", "Mês"], index=0)
    value = control_col_2.number_input("Quantidade", min_value=1.0, step=1.0, value=1.0, format="%.0f")

    returns_result = accumulated_returns_table(portfolio_df, historical_df, value, unit)
    if returns_result is not None:
        monthly_table, yearly_table, period_return, start_date, end_date = returns_result
        st.subheader("Rentabilidade acumulada")
        returns_df = _returns_table_to_frame(monthly_table, yearly_table)
        if not returns_df.empty:
            st.dataframe(
                returns_df,
                use_container_width=True,
                hide_index=True,
                column_config={label: st.column_config.NumberColumn(format="%.2f%%") for label in MONTH_LABELS + ["Acumulado (Ano)"]},
            )
        st.caption(
            f"Rentabilidade acumulada no periodo analisado ({start_date.strftime('%d/%m/%Y')} a {end_date.strftime('%d/%m/%Y')}): "
            f"{period_return:.2%}"
        )

    st.subheader("Analise individual dos investimentos")
    metrics = individual_metrics(portfolio_df, historical_df, value, unit)
    if metrics is None:
        st.info("Dados insuficientes para a analise individual.")
    else:
        metrics_df = pd.DataFrame(metrics).rename(index={"media": "Media", "variancia": "Variancia", "volatilidade": "Volatilidade"})
        st.dataframe(metrics_df, use_container_width=True)

    st.subheader("VaR e CVaR")
    var_95 = var_cvar_metrics(portfolio_df, historical_df, total_pl, value, unit, 0.95)
    var_99 = var_cvar_metrics(portfolio_df, historical_df, total_pl, value, unit, 0.99)
    var_col_1, var_col_2 = st.columns(2)
    with var_col_1:
        if var_95 is None:
            st.info("Nao foi possivel calcular o VaR/CVaR com 95%.")
        else:
            st.write("Nivel de confianca 95%")
            st.dataframe(pd.DataFrame(var_95).rename(index={"var": "VaR", "cvar": "CVaR"}), use_container_width=True)
    with var_col_2:
        if var_99 is None:
            st.info("Nao foi possivel calcular o VaR/CVaR com 99%.")
        else:
            st.write("Nivel de confianca 99%")
            st.dataframe(pd.DataFrame(var_99).rename(index={"var": "VaR", "cvar": "CVaR"}), use_container_width=True)

    st.subheader("Avaliacao de rendimento e risco")
    capm_metrics = capm_alpha_beta_correlation(portfolio_df, historical_df, value, unit)
    if capm_metrics is None:
        st.info("Dados insuficientes para CAPM, alfa, beta e correlacao.")
    else:
        metric_col_1, metric_col_2, metric_col_3, metric_col_4 = st.columns(4)
        metric_col_1.metric("CAPM", f"{capm_metrics['capm']:.2%}")
        metric_col_2.metric("Alfa", f"{capm_metrics['alfa']:.4f}")
        metric_col_3.metric("Beta", f"{capm_metrics['beta']:.4f}")
        metric_col_4.metric("Correlacao", f"{capm_metrics['correlacao']:.4f}")

    st.subheader("Indices de desempenho ajustados ao benchmark")
    indices = performance_indices(portfolio_df, historical_df, value, unit)
    if indices is None:
        st.info("Dados insuficientes para os indices de desempenho.")
    else:
        indices_df = pd.DataFrame(indices).T.rename(
            columns={
                "sharpe": "Sharpe",
                "sortino": "Sortino",
                "treynor": "Treynor",
                "tracking_error": "Tracking Error",
            }
        )
        st.dataframe(indices_df, use_container_width=True)

    st.subheader("Correlacao entre os ativos")
    correlation_result = correlation_matrix(portfolio_df, historical_df, value, unit)
    if correlation_result is None:
        st.info("Sao necessarios pelo menos dois ativos com historico valido para a matriz de correlacao.")
    else:
        matrix, _ = correlation_result
        st.dataframe(matrix.style.format("{:.4f}").background_gradient(cmap="Blues"), use_container_width=True)
