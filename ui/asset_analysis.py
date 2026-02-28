from __future__ import annotations

import pandas as pd
import streamlit as st

from services.market_data import AssetSnapshot, MarketDataError, MetricValue, fetch_asset_snapshot


def _render_metric_card(column, metric: MetricValue) -> None:
    with column:
        with st.container(border=True):
            st.metric(metric.label, metric.formatted())
            if not metric.is_available:
                st.caption(metric.unavailable_reason)


def _render_section(title: str, metrics: list[MetricValue]) -> None:
    st.subheader(title)
    columns = st.columns(len(metrics))
    for column, metric in zip(columns, metrics):
        _render_metric_card(column, metric)


def _render_details(snapshot: AssetSnapshot) -> None:
    with st.expander("Detalhes retornados pelo Yahoo Finance"):
        st.dataframe(snapshot.details, use_container_width=True, hide_index=True)


def _normalize_metric_history(history_df: pd.DataFrame) -> pd.DataFrame:
    normalized = pd.DataFrame(index=history_df.index)
    for column in history_df.columns:
        series = pd.to_numeric(history_df[column], errors="coerce")
        valid = series.dropna()
        if valid.empty:
            continue
        base_value = float(valid.iloc[0])
        if base_value == 0:
            normalized[column] = series
            continue
        normalized[column] = series / base_value * 100
    return normalized.dropna(how="all")


def _render_metric_history(snapshot: AssetSnapshot) -> None:
    st.subheader("Histórico dos Indicadores")
    if snapshot.metric_history.empty:
        st.info("O Yahoo Finance não retornou histórico fundamentalista suficiente para montar os indicadores.")
        return

    available_metrics = snapshot.metric_history.dropna(axis=1, how="all").columns.tolist()
    default_metrics = [
        metric
        for metric in ["market_cap", "enterprise_value", "pe_ratio", "ev_to_ebitda", "roic"]
        if metric in available_metrics
    ]
    if not default_metrics:
        default_metrics = available_metrics[:4]

    selected_metrics = st.multiselect(
        "Indicadores para comparar",
        options=available_metrics,
        default=default_metrics,
        format_func=lambda metric: snapshot.metric_labels.get(metric, metric),
    )
    if not selected_metrics:
        st.info("Selecione pelo menos um indicador para exibir o gráfico histórico.")
        return

    compare_mode = st.toggle("Normalizar séries para comparação", value=True)

    chart_df = snapshot.metric_history[selected_metrics].copy()
    if compare_mode:
        chart_df = _normalize_metric_history(chart_df)
        st.caption("Séries normalizadas em 100 no primeiro ponto disponível para facilitar a comparação.")

    chart_df = chart_df.rename(columns=snapshot.metric_labels)
    st.line_chart(chart_df, use_container_width=True)

    with st.expander("Tabela histórica dos indicadores"):
        raw_history = snapshot.metric_history[selected_metrics].rename(columns=snapshot.metric_labels).copy()
        raw_history.index = pd.to_datetime(raw_history.index, errors="coerce")
        st.dataframe(raw_history, use_container_width=True)


def render_asset_analysis_page(default_ticker: str = "") -> None:
    st.title("Análise de Ativos")
    st.caption("Consulta de indicadores fundamentalistas e de mercado via Yahoo Finance.")

    with st.form("asset-analysis-form", clear_on_submit=False):
        ticker = st.text_input(
            "Ticker",
            value=default_ticker,
            placeholder="Ex.: AAPL, MSFT, PETR4.SA, VALE3.SA",
        )
        submitted = st.form_submit_button("Analisar", use_container_width=True)

    if not submitted and not default_ticker:
        st.info("Informe um ticker e clique em 'Analisar'.")
        return

    if not ticker.strip():
        st.warning("Digite um ticker válido para iniciar a consulta.")
        return

    try:
        with st.spinner("Consultando Yahoo Finance..."):
            snapshot = fetch_asset_snapshot(ticker)
    except MarketDataError as exc:
        st.error(str(exc))
        return
    except Exception as exc:
        st.error(f"Erro inesperado ao montar a análise do ativo: {exc}")
        return

    st.success(f"{snapshot.long_name} ({snapshot.ticker})")
    st.caption(f"Moeda reportada pelo Yahoo Finance: {snapshot.currency}")

    _render_section(
        "Preço & Mercado",
        [
            snapshot.metrics["current_price"],
            snapshot.metrics["weekly_price_change_pct"],
            snapshot.metrics["market_cap"],
            snapshot.metrics["enterprise_value"],
        ],
    )
    _render_section(
        "Valuation",
        [
            snapshot.metrics["roic"],
            snapshot.metrics["pe_ratio"],
            snapshot.metrics["dividend_yield"],
            snapshot.metrics["ev_to_ebitda"],
        ],
    )
    _render_section(
        "Qualidade & Risco",
        [
            snapshot.metrics["ebitda_margin"],
            snapshot.metrics["financial_leverage"],
            snapshot.metrics["beta"],
        ],
    )

    _render_metric_history(snapshot)

    st.caption("Quando um campo aparece como N/A, o dado não foi disponibilizado pelo Yahoo Finance para este ticker.")
    _render_details(snapshot)
