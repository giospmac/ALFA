from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from services.market_data import AssetSnapshot, MarketDataError, MetricValue, fetch_asset_snapshot


def _render_asset_analysis_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --asset-border: #cfd4de;
            --asset-radius: 18px;
        }
        .stApp {
            background: #f4f5f0;
        }
        .stApp h1,
        .stApp h2,
        .stApp h3,
        .stApp strong,
        .stMetric label,
        .stMetric [data-testid="stMetricValue"],
        [data-testid="stMarkdownContainer"] p strong {
            color: #4979f6;
        }
        .stApp p,
        .stApp label,
        .stApp .stCaption,
        .stApp [data-testid="stCaptionContainer"],
        .stApp [data-testid="stMarkdownContainer"] p,
        .stApp [data-testid="stMarkdownContainer"] li {
            color: #717171;
        }
        [data-testid="stTextInput"] label,
        [data-testid="stTextInput"] input,
        [data-testid="stTextInput"] input::placeholder {
            color: #717171;
        }
        [data-testid="stMultiSelect"] label,
        [data-testid="stSelectbox"] label,
        [data-testid="stTextInput"] label {
            color: #717171;
        }
        [data-testid="stTextInput"] input {
            border-color: rgba(113, 113, 113, 0.25);
        }
        [data-testid="stVerticalBlockBorderWrapper"]:has(.stMetric) {
            border: 1px solid var(--asset-border) !important;
            border-radius: var(--asset-radius) !important;
            background: #f4f5f0 !important;
        }
        [data-testid="stVerticalBlockBorderWrapper"]:has(.asset-analysis-shell) {
            border: 1px solid var(--asset-border) !important;
            border-radius: var(--asset-radius) !important;
            background: #f4f5f0 !important;
        }
        [data-testid="stVerticalBlockBorderWrapper"]:has([data-testid="stVegaLiteChart"]) {
            border: 1px solid var(--asset-border) !important;
            border-radius: var(--asset-radius) !important;
            background: #f4f5f0 !important;
            padding: 0.35rem 0.35rem 0.15rem 0.35rem;
        }
        [data-testid="stVegaLiteChart"],
        [data-testid="stVegaLiteChart"] > div {
            background: #f4f5f0 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


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


def _render_history_chart(chart_df: pd.DataFrame, label: str) -> None:
    plot_df = chart_df.reset_index().rename(columns={chart_df.index.name or "index": "Data", label: "Valor"})
    chart = (
        alt.Chart(plot_df)
        .mark_line(color="#4979f6", strokeWidth=2.5)
        .encode(
            x=alt.X("Data:T", title=None, axis=alt.Axis(labelColor="#717171", domainColor="#dddddd", tickColor="#dddddd")),
            y=alt.Y("Valor:Q", title=None, axis=alt.Axis(labelColor="#717171", domainColor="#dddddd", tickColor="#dddddd")),
            tooltip=[alt.Tooltip("Data:T", title="Data"), alt.Tooltip("Valor:Q", title=label, format=",.4f")],
        )
        .properties(height=240)
        .configure(background="#f4f5f0")
        .configure_view(stroke="#cfd4de", cornerRadius=18)
        .configure_axis(gridColor="#dddddd", gridOpacity=0.35)
    )
    st.altair_chart(chart, use_container_width=True)


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
            with st.container(border=True):
                st.markdown(f"**{label}**")
                _render_history_chart(chart_df, label)
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
    _render_asset_analysis_styles()
    st.title("Análise de Ativos")
    st.caption("Consulta de indicadores fundamentalistas e de mercado via Yahoo Finance.")

    shell = st.container(border=True)
    with shell:
        st.markdown('<div class="asset-analysis-shell"></div>', unsafe_allow_html=True)
        with st.form("asset-analysis-form", clear_on_submit=False):
            ticker = st.text_input(
                "Ticker",
                value=default_ticker,
                placeholder="Ex.: AAPL, MSFT, PETR4.SA, VALE3.SA",
            )
            submitted = st.form_submit_button("Analisar", use_container_width=True)

    if not submitted and not default_ticker:
        shell.info("Informe um ticker e clique em 'Analisar'.")
        return

    if not ticker.strip():
        shell.warning("Digite um ticker válido para iniciar a consulta.")
        return

    try:
        with st.spinner("Consultando Yahoo Finance..."):
            snapshot = fetch_asset_snapshot(ticker)
    except MarketDataError as exc:
        shell.error(str(exc))
        return
    except Exception as exc:
        shell.error(f"Erro inesperado ao montar a análise do ativo: {exc}")
        return

    shell.success(f"{snapshot.long_name} ({snapshot.ticker})")
    shell.caption(f"Moeda reportada pelo Yahoo Finance: {snapshot.currency}")

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
