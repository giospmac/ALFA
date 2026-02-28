from __future__ import annotations

import pandas as pd
import streamlit as st

from services.market_data import AssetSnapshot, MarketDataError, MetricValue, fetch_asset_snapshot


def _reference_note(snapshot: AssetSnapshot, metric_key: str) -> str:
    reference_df = snapshot.metric_reference
    selected = reference_df[reference_df["key"] == metric_key]
    if selected.empty:
        return ""
    row = selected.iloc[0]
    return f'Periodo: {row["Periodo de referencia"]} | Data-base: {row["Data de referência"]}'


def _render_metric_card(column, metric: MetricValue, reference_note: str) -> None:
    with column:
        with st.container(border=True):
            st.metric(metric.label, metric.formatted())
            if not metric.is_available:
                st.caption(metric.unavailable_reason)
            if reference_note:
                st.caption(reference_note)


def _render_section(snapshot: AssetSnapshot, title: str, metric_keys: list[str]) -> None:
    st.subheader(title)
    columns = st.columns(len(metric_keys))
    for column, metric_key in zip(columns, metric_keys):
        _render_metric_card(column, snapshot.metrics[metric_key], _reference_note(snapshot, metric_key))


def _render_details(snapshot: AssetSnapshot) -> None:
    with st.expander("Detalhes retornados pelo Yahoo Finance"):
        st.dataframe(snapshot.details, use_container_width=True, hide_index=True)


def _render_metric_history(snapshot: AssetSnapshot) -> None:
    st.subheader("Histórico dos Indicadores")
    reference_df = snapshot.metric_reference.copy()
    history_keys = reference_df.loc[reference_df["Histórico disponível"] == "Sim", "key"].tolist()

    if snapshot.metric_history.empty or not history_keys:
        st.info("Entre os indicadores solicitados, o Yahoo Finance não retornou séries históricas suficientes para gráfico.")
        return

    display_metrics = [
        metric
        for metric in ["market_cap", "enterprise_value", "ebitda_margin", "ev_to_ebitda", "roic", "dividend_yield", "pe_ratio", "financial_leverage", "weekly_price_change_pct"]
        if metric in history_keys and metric in snapshot.metric_history.columns
    ]
    if not display_metrics:
        return

    columns = st.columns(2, gap="large")
    for index, metric in enumerate(display_metrics):
        label = snapshot.metric_labels.get(metric, metric)
        chart_df = snapshot.metric_history[[metric]].dropna().rename(columns={metric: label})
        if chart_df.empty:
            continue
        with columns[index % 2]:
            st.markdown(f"**{label}**")
            st.line_chart(chart_df, use_container_width=True, color=["#4979f6"])
            st.caption(_reference_note(snapshot, metric))

    with st.expander("Tabela histórica dos indicadores"):
        raw_history = snapshot.metric_history[display_metrics].rename(columns=snapshot.metric_labels).copy()
        st.dataframe(raw_history, use_container_width=True)


def _render_metric_reference(snapshot: AssetSnapshot) -> None:
    st.subheader("Referência dos Indicadores")
    st.caption("Indicadores sem histórico suficiente ficam apenas com o último valor disponível e a respectiva data-base.")

    reference_df = snapshot.metric_reference.copy()
    without_history = reference_df[reference_df["Histórico disponível"] == "Não"].drop(columns=["key"])
    if without_history.empty:
        st.caption("Todos os indicadores exibidos acima possuem alguma série histórica utilizável no Yahoo Finance.")
    else:
        st.dataframe(without_history, use_container_width=True, hide_index=True)

    with st.expander("Mapa completo de disponibilidade"):
        st.dataframe(reference_df.drop(columns=["key"]), use_container_width=True, hide_index=True)


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
        snapshot,
        "Preço & Mercado",
        ["current_price", "weekly_price_change_pct", "market_cap", "enterprise_value"],
    )
    _render_section(
        snapshot,
        "Valuation",
        ["roic", "pe_ratio", "dividend_yield", "ev_to_ebitda"],
    )
    _render_section(
        snapshot,
        "Qualidade & Risco",
        ["ebitda_margin", "financial_leverage", "beta"],
    )

    _render_metric_history(snapshot)
    _render_metric_reference(snapshot)

    st.caption("Quando um campo aparece como N/A, o dado não foi disponibilizado pelo Yahoo Finance para este ticker.")
    _render_details(snapshot)
