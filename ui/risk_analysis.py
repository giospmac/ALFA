from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.colors as mcolors

from core.config_repository import ConfigRepository
from core.portfolio_repository import PortfolioRepository
from services.capm import (
    capm_expected_return,
    capm_per_asset_table,
    estimate_beta,
    jensen_alpha,
    realized_market_premium,
    systematic_idiosyncratic_risk,
)
from services.portfolio_analytics import (
    ANNUAL_CDI_RATE,
    TRADING_DAYS_PER_YEAR,
    _active_assets,
    _available_asset_columns,
    _normalized_weights,
    _portfolio_log_returns,
    accumulated_returns_table,
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
    st.caption("Tabelas de rentabilidade, VaR, CVaR, CAPM forward-looking, decomposição de risco e correlação.")

    if portfolio_df.empty or historical_df.empty:
        st.info("Monte o portfolio e atualize o historico na pagina principal antes de abrir esta analise.")
        return

    min_date = historical_df.index.min().date() if not historical_df.empty else datetime.today().date()
    max_date = historical_df.index.max().date() if not historical_df.empty else datetime.today().date()
    
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

    # ── Accumulated returns ──────────────────────────────────────
    returns_result = accumulated_returns_table(portfolio_df, historical_df, start_ts, end_ts)
    if returns_result is not None:
        monthly_table, yearly_table, period_return, start_date, end_date = returns_result
        st.subheader("Rentabilidade acumulada")
        returns_df = _returns_table_to_frame(monthly_table, yearly_table)
        if not returns_df.empty:
            custom_cmap = mcolors.LinearSegmentedColormap.from_list(
                "custom_blues", ["#97bdff", "#4979f6", "#102170"]
            )
            st.dataframe(
                returns_df.style
                .format("{:.2f}%", na_rep="-", subset=MONTH_LABELS + ["Acumulado (Ano)"])
                .format("{:.0f}", na_rep="-", subset=["Ano"])
                .highlight_null(color="rgba(0,0,0,0)")
                .background_gradient(
                    subset=MONTH_LABELS + ["Acumulado (Ano)"], cmap=custom_cmap
                )
                .set_properties(**{"text-align": "center"}),
                use_container_width=True,
                hide_index=True,
            )
        st.caption(
            f"Rentabilidade acumulada no período analisado ({start_date.strftime('%d/%m/%Y')} a {end_date.strftime('%d/%m/%Y')}): "
            f"{period_return:.2%}"
        )

    # ── Individual metrics ───────────────────────────────────────
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

    # ── VaR / CVaR ───────────────────────────────────────────────
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

    # ── CAPM Forward-Looking ─────────────────────────────────────
    st.subheader("Avaliação de rendimento e risco (CAPM)")

    # Premisses block — persisted via config
    config_repo = ConfigRepository(base_path=Path(__file__).resolve().parents[1])
    config = config_repo.load()

    with st.expander("⚙️  Premissas CAPM", expanded=False):
        prem_col_1, prem_col_2 = st.columns(2)
        rf_input = prem_col_1.number_input(
            "Taxa livre de risco (a.a.)",
            value=config.risk_free_rate,
            min_value=0.0,
            max_value=1.0,
            step=0.0025,
            format="%.4f",
            help="CDI ou Selic anual. Usado como Rf em todos os modelos.",
        )
        emrp_input = prem_col_2.number_input(
            "Prêmio de risco esperado do mercado (EMRP a.a.)",
            value=config.emrp,
            min_value=0.0,
            max_value=0.30,
            step=0.005,
            format="%.4f",
            help="E(Rm) − Rf. Premissa forward-looking para o retorno esperado do mercado.",
        )
        if rf_input != config.risk_free_rate or emrp_input != config.emrp:
            config_repo.update(risk_free_rate=rf_input, emrp=emrp_input)

    rf_annual = rf_input
    emrp_annual = emrp_input
    rf_daily = (1 + rf_annual) ** (1 / TRADING_DAYS_PER_YEAR) - 1

    # Portfolio-level CAPM
    period_df = historical_df.loc[start_ts:end_ts].ffill().bfill()
    if not period_df.empty and "IBOVESPA" in period_df.columns:
        log_returns = np.log(period_df / period_df.shift(1)).dropna(how="all")
        portfolio_returns = _portfolio_log_returns(portfolio_df, period_df)
        market_ret = log_returns.get("IBOVESPA", pd.Series(dtype=float)).dropna()

        if not portfolio_returns.empty and not market_ret.empty and len(portfolio_returns) > 20:
            beta_val = estimate_beta(portfolio_returns, market_ret)
            capm_er = capm_expected_return(rf_annual, beta_val, emrp_annual)
            alpha_val = jensen_alpha(portfolio_returns, market_ret, rf_daily)
            realized_prem = realized_market_premium(market_ret, rf_daily)
            risk = systematic_idiosyncratic_risk(portfolio_returns, market_ret, beta_val)
            corr_val = float(
                pd.concat([portfolio_returns.rename("p"), market_ret.rename("m")], axis=1)
                .dropna()
                .corr()
                .loc["p", "m"]
            )

            # KPI cards row 1
            kpi_1, kpi_2, kpi_3, kpi_4 = st.columns(4)
            kpi_1.metric("CAPM Esperado (forward)", f"{capm_er:.2%}")
            kpi_2.metric("Alfa de Jensen", f"{alpha_val:.4f}")
            kpi_3.metric("Beta", f"{beta_val:.4f}")
            kpi_4.metric("Correlação (IBOV)", f"{corr_val:.4f}")

            # KPI cards row 2 — premiums & risk decomposition
            kpi_5, kpi_6, kpi_7, kpi_8 = st.columns(4)
            kpi_5.metric("Prêmio Realizado (hist.)", f"{realized_prem:.2%}")
            kpi_6.metric("Prêmio Esperado (EMRP)", f"{emrp_annual:.2%}")
            kpi_7.metric("Vol. Sistêmica", f"{risk.systematic_vol_annual:.2%}")
            kpi_8.metric("Vol. Idiossincrática", f"{risk.idiosyncratic_vol_annual:.2%}")

            # Row 3
            kpi_9, kpi_10, _, _ = st.columns(4)
            kpi_9.metric("Vol. Total (anual)", f"{risk.total_vol_annual:.2%}")
            kpi_10.metric("R² (variância explicada)", f"{risk.r_squared:.2%}")
        else:
            st.info("Dados insuficientes para CAPM da carteira.")
    else:
        st.info("Dados insuficientes para CAPM, alfa, beta e correlação.")

    # ── Per-asset CAPM table ─────────────────────────────────────
    st.subheader("CAPM por ativo — decomposição de risco")
    asset_table = capm_per_asset_table(portfolio_df, historical_df, rf_annual, emrp_annual, start_ts, end_ts)
    if asset_table is not None and not asset_table.empty:
        pct_cols = [
            "Retorno Histórico", "CAPM Esperado", "Prêmio Realizado", "Prêmio Esperado",
            "Vol Total", "Vol Sistêmica", "Vol Idiossincrática", "R²",
        ]
        format_dict = {col: "{:.2%}" for col in pct_cols}
        format_dict["Beta"] = "{:.4f}"
        format_dict["Alfa de Jensen"] = "{:.4f}"
        st.dataframe(
            asset_table.style.format(format_dict).background_gradient(
                subset=["R²"], cmap="Blues", vmin=0, vmax=1
            ),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("São necessários pelo menos 20 observações por ativo para a tabela CAPM.")

    # ── Performance indices ──────────────────────────────────────
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

    # ── Correlation matrix ───────────────────────────────────────
    st.subheader("Matriz de Correlação Diária")
    correlation_result = correlation_matrix(portfolio_df, historical_df, start_ts, end_ts)
    if correlation_result is None:
        st.info("São necessários pelo menos dois ativos com histórico válido para a matriz de correlação.")
    else:
        matrix, _ = correlation_result
        st.dataframe(
            matrix.style.format("{:.4f}").background_gradient(cmap="RdBu", vmin=-1, vmax=1), 
            use_container_width=True
        )
