from __future__ import annotations

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


def _render_metric_history(snapshot: AssetSnapshot) -> None:
    st.subheader("Histórico dos Indicadores")
    reference_df = snapshot.metric_reference.copy()
    history_keys = reference_df.loc[reference_df["Histórico disponível"] == "Sim", "key"].tolist()

    if snapshot.metric_history.empty or not history_keys:
        st.info("Entre os indicadores solicitados, o Yahoo Finance não retornou séries históricas suficientes para gráfico.")
        return

    available_metrics = [metric for metric in history_keys if metric in snapshot.metric_history.columns]
    default_metrics = [
        metric
        for metric in ["market_cap", "enterprise_value", "ebitda_margin", "ev_to_ebitda", "roic"]
        if metric in available_metrics
    ]
    if not default_metrics:
        default_metrics = available_metrics[:4]

    selected_metrics = st.multiselect(
        "Indicadores para exibir",
        options=available_metrics,
        default=default_metrics,
        format_func=lambda metric: snapshot.metric_labels.get(metric, metric),
    )
    if not selected_metrics:
        st.info("Selecione pelo menos um indicador para exibir o gráfico histórico.")
        return

    for metric in selected_metrics:
        label = snapshot.metric_labels.get(metric, metric)
        chart_df = snapshot.metric_history[[metric]].dropna().rename(columns={metric: label})
        if chart_df.empty:
            continue
        st.markdown(f"**{label}**")
        st.line_chart(chart_df, use_container_width=True)

    with st.expander("Tabela histórica dos indicadores"):
        raw_history = snapshot.metric_history[selected_metrics].rename(columns=snapshot.metric_labels).copy()
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
    _render_metric_reference(snapshot)

    st.caption("Quando um campo aparece como N/A, o dado não foi disponibilizado pelo Yahoo Finance para este ticker.")
    _render_details(snapshot)
