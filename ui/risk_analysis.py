from __future__ import annotations

from datetime import datetime, timedelta
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
    st.caption("Tabelas de rentabilidade, VaR, CVaR, CAPM, performance ajustada ao benchmark e correlação.")

    if portfolio_df.empty or historical_df.empty:
        st.info("Monte o portfolio e atualize o historico na pagina principal antes de abrir esta analise.")
        return

    min_date = historical_df.index.min().date() if not historical_df.empty else datetime.today().date()
    max_date = historical_df.index.max().date() if not historical_df.empty else datetime.today().date()
    
    # Date picker logic handling edge cases with minimal dates
    default_start = max_date - timedelta(days=365)
    if default_start < min_date:
        default_start = min_date

    control_col_1, control_col_2, _ = st.columns([1.5, 1.5, 1])
    start_date_val = control_col_1.date_input("Data Inicial", value=default_start, min_value=min_date, max_value=max_date)
    end_date_val = control_col_2.date_input("Data Final", value=max_date, min_value=min_date, max_value=max_date)
    
    start_ts = pd.Timestamp(start_date_val)
    end_ts = pd.Timestamp(end_date_val)
    
    if start_ts > end_ts:
        st.error("A Data Inicial não pode ser maior do que a Data Final.")
        return

    st.markdown("---")

    returns_result = accumulated_returns_table(portfolio_df, historical_df, start_ts, end_ts)
    if returns_result is not None:
        monthly_table, yearly_table, period_return, start_date, end_date = returns_result
        st.subheader("Rentabilidade acumulada")
        returns_df = _returns_table_to_frame(monthly_table, yearly_table)
        if not returns_df.empty:
            st.dataframe(
                returns_df.style.format("{:.2f}%", na_rep="-").background_gradient(subset=MONTH_LABELS + ["Acumulado (Ano)"], cmap="RdYlGn"),
                use_container_width=True,
                hide_index=True,
            )
        st.caption(
            f"Rentabilidade acumulada no período analisado ({start_date.strftime('%d/%m/%Y')} a {end_date.strftime('%d/%m/%Y')}): "
            f"{period_return:.2%}"
        )

    st.subheader("Análise individual dos investimentos")
    metrics = individual_metrics(portfolio_df, historical_df, start_ts, end_ts)
    if metrics is None:
        st.info("Dados insuficientes para a analise individual.")
    else:
        metrics_df = pd.DataFrame(metrics).rename(index={"media": "Média D.", "variancia": "Variância D.", "volatilidade": "Volatilidade D."})
        st.dataframe(
            metrics_df.style.format("{:.4f}").background_gradient(cmap="Blues", axis=1), 
            use_container_width=True
        )

    st.subheader("VaR e CVaR (Valor em Risco)")
    var_95 = var_cvar_metrics(portfolio_df, historical_df, total_pl, start_ts, end_ts, 0.95)
    var_99 = var_cvar_metrics(portfolio_df, historical_df, total_pl, start_ts, end_ts, 0.99)
    var_col_1, var_col_2 = st.columns(2)
    with var_col_1:
        if var_95 is None:
            st.info("Não foi possível calcular o VaR/CVaR a 95%.")
        else:
            st.write("Nível de confiança 95%")
            var_95_df = pd.DataFrame(var_95).rename(index={"var": "VaR (R$)", "cvar": "CVaR (R$)"})
            st.dataframe(var_95_df.style.format("R$ {:,.2f}").background_gradient(cmap="Reds", axis=0), use_container_width=True)
    with var_col_2:
        if var_99 is None:
            st.info("Não foi possível calcular o VaR/CVaR a 99%.")
        else:
            st.write("Nível de confiança 99%")
            var_99_df = pd.DataFrame(var_99).rename(index={"var": "VaR (R$)", "cvar": "CVaR (R$)"})
            st.dataframe(var_99_df.style.format("R$ {:,.2f}").background_gradient(cmap="Reds", axis=0), use_container_width=True)

    st.subheader("Avaliação de rendimento e risco (CAPM)")
    capm_metrics = capm_alpha_beta_correlation(portfolio_df, historical_df, start_ts, end_ts)
    if capm_metrics is None:
        st.info("Dados insuficientes para CAPM, alfa, beta e correlação.")
    else:
        metric_col_1, metric_col_2, metric_col_3, metric_col_4 = st.columns(4)
        metric_col_1.metric("CAPM", f"{capm_metrics['capm']:.2%}")
        metric_col_2.metric("Alfa de Jensen", f"{capm_metrics['alfa']:.4f}")
        metric_col_3.metric("Beta", f"{capm_metrics['beta']:.4f}")
        metric_col_4.metric("Correlação (IBOV)", f"{capm_metrics['correlacao']:.4f}")

    st.subheader("Índices de desempenho ajustados ao benchmark")
    indices = performance_indices(portfolio_df, historical_df, start_ts, end_ts)
    if indices is None:
        st.info("Dados insuficientes para os índices de desempenho.")
    else:
        indices_df = pd.DataFrame(indices).T.rename(
            columns={
                "sharpe": "Sharpe",
                "sortino": "Sortino",
                "treynor": "Treynor",
                "tracking_error": "Tracking Error",
            }
        )
        st.dataframe(
            indices_df.style.format("{:.4f}").background_gradient(cmap="Blues", axis=0), 
            use_container_width=True
        )

    st.subheader("Matriz de Correlação Diária")
    correlation_result = correlation_matrix(portfolio_df, historical_df, start_ts, end_ts)
    if correlation_result is None:
        st.info("São necessários pelo menos dois ativos com histórico válido para a matriz de correlação.")
    else:
        matrix, _ = correlation_result
        # The correlation is usually formatted from -1 to 1 (RdBu cmap is better for correlations!)
        st.dataframe(
            matrix.style.format("{:.4f}").background_gradient(cmap="RdBu", vmin=-1, vmax=1), 
            use_container_width=True
        )
