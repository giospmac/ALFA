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
    except Exception:
        st.error("Ocorreu um erro inesperado ao montar a análise do ativo.")
        return

    st.success(f"{snapshot.long_name} ({snapshot.ticker})")
    st.caption(f"Moeda reportada pelo Yahoo Finance: {snapshot.currency}")

    _render_section(
        "Preço & Risco",
        [
            snapshot.metrics["current_price"],
            snapshot.metrics["beta"],
            snapshot.metrics["day_change"],
            snapshot.metrics["day_change_pct"],
        ],
    )
    _render_section(
        "Valuation",
        [
            snapshot.metrics["pe_ratio"],
            snapshot.metrics["ev_to_ebitda"],
            snapshot.metrics["market_cap"],
        ],
    )
    _render_section(
        "Operação",
        [
            snapshot.metrics["ebitda_margin"],
        ],
    )
    _render_section(
        "Crescimento",
        [
            snapshot.metrics["revenue_growth_yoy"],
        ],
    )

    st.caption("Quando um campo aparece como N/A, o dado não foi disponibilizado pelo Yahoo Finance para este ticker.")
    _render_details(snapshot)
