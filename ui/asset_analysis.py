from __future__ import annotations

from html import escape

import pandas as pd
import streamlit as st

from services.market_data import AssetSnapshot, MarketDataError, MetricValue, fetch_asset_snapshot


def _render_asset_analysis_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --asset-page-bg: #f6f4ef;
            --asset-surface: rgba(255, 255, 255, 0.78);
            --asset-surface-strong: rgba(255, 255, 255, 0.92);
            --asset-border: rgba(21, 33, 53, 0.10);
            --asset-border-strong: rgba(21, 33, 53, 0.16);
            --asset-text: #172033;
            --asset-muted: #6c7483;
            --asset-soft: #8c93a3;
            --asset-accent: #2f5a7a;
            --asset-accent-soft: rgba(47, 90, 122, 0.10);
            --asset-radius-lg: 28px;
            --asset-radius-md: 22px;
            --asset-radius-sm: 18px;
            --asset-shadow: 0 18px 38px rgba(23, 32, 51, 0.06);
        }
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(210, 220, 228, 0.55), transparent 28%),
                radial-gradient(circle at top right, rgba(235, 230, 220, 0.8), transparent 34%),
                linear-gradient(180deg, #faf8f4 0%, var(--asset-page-bg) 42%, #f3f0e8 100%);
            color: var(--asset-text);
        }
        section.main [data-testid="block-container"] {
            max-width: 1220px;
            padding-top: 2.3rem;
            padding-bottom: 3.5rem;
        }
        section.main .block-container h1,
        section.main .block-container h2,
        section.main .block-container h3,
        section.main .block-container strong {
            color: var(--asset-text);
        }
        section.main .block-container p,
        section.main .block-container label,
        section.main .block-container .stCaption,
        section.main .block-container [data-testid="stCaptionContainer"],
        section.main .block-container [data-testid="stMarkdownContainer"] p,
        section.main .block-container [data-testid="stMarkdownContainer"] li {
            color: var(--asset-muted);
        }
        section.main div[data-testid="stTextInput"] label {
            color: var(--asset-muted);
            font-size: 0.92rem;
            font-weight: 600;
        }
        section.main div[data-testid="stTextInput"] input {
            border-radius: 16px;
            border: 1px solid var(--asset-border);
            background: rgba(255, 255, 255, 0.88);
            color: var(--asset-text);
            min-height: 3.2rem;
            box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.55);
        }
        section.main div[data-testid="stTextInput"] input::placeholder {
            color: #9aa1ad;
        }
        section.main div[data-testid="stTextInput"] input:focus {
            border-color: rgba(47, 90, 122, 0.28);
            box-shadow: 0 0 0 1px rgba(47, 90, 122, 0.10);
        }
        section.main div[data-testid="stFormSubmitButton"] > button,
        section.main div[data-testid="stButton"] > button {
            border-radius: 999px;
            min-height: 3rem;
            border: 1px solid var(--asset-border);
            background: rgba(255, 255, 255, 0.68);
            color: var(--asset-text);
            box-shadow: none;
            transition: all 0.18s ease;
        }
        section.main div[data-testid="stFormSubmitButton"] > button:hover,
        section.main div[data-testid="stButton"] > button:hover {
            border-color: rgba(47, 90, 122, 0.24);
            background: rgba(255, 255, 255, 0.92);
            color: var(--asset-text);
        }
        section.main div[data-testid="stFormSubmitButton"] > button[kind="primary"],
        section.main div[data-testid="stButton"] > button[kind="primary"] {
            border-color: rgba(47, 90, 122, 0.30);
            background: linear-gradient(180deg, #23374a 0%, #1c2f40 100%);
            color: #f9fafb;
        }
        [data-testid="stVerticalBlockBorderWrapper"]:has(.asset-analysis-search-shell) {
            border: 1px solid var(--asset-border) !important;
            border-radius: var(--asset-radius-lg) !important;
            background: var(--asset-surface) !important;
            backdrop-filter: blur(10px);
            box-shadow: var(--asset-shadow);
            padding: 1.25rem 1.25rem 0.85rem 1.25rem;
        }
        .asset-page-hero {
            position: relative;
            overflow: hidden;
            border-radius: 30px;
            border: 1px solid var(--asset-border);
            background:
                linear-gradient(135deg, rgba(255, 255, 255, 0.70), rgba(255, 255, 255, 0.38)),
                radial-gradient(circle at top right, rgba(47, 90, 122, 0.10), transparent 35%);
            padding: 1.7rem 1.75rem;
            margin-bottom: 1.2rem;
            box-shadow: var(--asset-shadow);
            backdrop-filter: blur(12px);
        }
        .asset-page-kicker {
            color: var(--asset-soft);
            font-size: 0.73rem;
            font-weight: 700;
            letter-spacing: 0.18em;
            text-transform: uppercase;
            margin-bottom: 0.7rem;
        }
        .asset-page-title {
            color: var(--asset-text);
            font-size: clamp(2rem, 3.1vw, 3.35rem);
            line-height: 0.98;
            font-weight: 700;
            letter-spacing: -0.03em;
            margin-bottom: 0.7rem;
        }
        .asset-page-subtitle {
            max-width: 760px;
            color: var(--asset-muted);
            font-size: 1rem;
            line-height: 1.65;
        }
        .asset-panel-kicker {
            color: var(--asset-soft);
            font-size: 0.74rem;
            font-weight: 700;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            margin-bottom: 0.45rem;
        }
        .asset-panel-title {
            color: var(--asset-text);
            font-size: 1.22rem;
            font-weight: 600;
            margin-bottom: 0.25rem;
        }
        .asset-panel-copy {
            color: var(--asset-muted);
            font-size: 0.95rem;
            margin-bottom: 1rem;
        }
        .asset-empty-state {
            border: 1px dashed rgba(23, 32, 51, 0.16);
            border-radius: 22px;
            padding: 1rem 1.05rem;
            margin-top: 0.65rem;
            background: rgba(255, 255, 255, 0.40);
            color: var(--asset-muted);
        }
        .asset-summary-card {
            border: 1px solid var(--asset-border);
            border-radius: 30px;
            background: var(--asset-surface-strong);
            padding: 1.5rem 1.55rem;
            margin: 1.15rem 0 1.2rem 0;
            box-shadow: var(--asset-shadow);
        }
        .asset-summary-kicker {
            color: var(--asset-soft);
            font-size: 0.74rem;
            font-weight: 700;
            letter-spacing: 0.16em;
            text-transform: uppercase;
            margin-bottom: 0.55rem;
        }
        .asset-summary-header {
            display: flex;
            justify-content: space-between;
            gap: 1rem;
            align-items: flex-start;
            flex-wrap: wrap;
            margin-bottom: 1rem;
        }
        .asset-summary-name {
            color: var(--asset-text);
            font-size: clamp(1.7rem, 2.3vw, 2.5rem);
            line-height: 1.05;
            font-weight: 700;
            letter-spacing: -0.03em;
            margin-bottom: 0.28rem;
        }
        .asset-summary-code {
            color: var(--asset-muted);
            font-size: 0.96rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }
        .asset-summary-price {
            color: var(--asset-text);
            font-size: clamp(1.7rem, 2.4vw, 2.35rem);
            line-height: 1;
            font-weight: 700;
            letter-spacing: -0.03em;
            text-align: right;
        }
        .asset-summary-change {
            margin-top: 0.45rem;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            border-radius: 999px;
            padding: 0.38rem 0.78rem;
            font-size: 0.86rem;
            font-weight: 700;
            letter-spacing: 0.02em;
        }
        .asset-summary-change.positive {
            background: rgba(34, 117, 78, 0.10);
            color: #1f6b4b;
        }
        .asset-summary-change.negative {
            background: rgba(163, 62, 62, 0.10);
            color: #9c3131;
        }
        .asset-summary-change.neutral {
            background: rgba(23, 32, 51, 0.06);
            color: var(--asset-muted);
        }
        .asset-pill-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.55rem;
            margin-top: 1rem;
        }
        .asset-pill {
            display: inline-flex;
            align-items: center;
            border-radius: 999px;
            padding: 0.5rem 0.78rem;
            background: var(--asset-accent-soft);
            color: var(--asset-accent);
            font-size: 0.84rem;
            font-weight: 600;
        }
        .asset-section-heading {
            margin: 1.6rem 0 0.9rem 0;
        }
        .asset-section-title {
            color: var(--asset-text);
            font-size: 1.26rem;
            font-weight: 700;
            letter-spacing: -0.02em;
            margin-bottom: 0.28rem;
        }
        .asset-section-copy {
            color: var(--asset-muted);
            font-size: 0.94rem;
            line-height: 1.55;
        }
        .asset-metric-card {
            min-height: 182px;
            border: 1px solid var(--asset-border);
            border-radius: 24px;
            background: rgba(255, 255, 255, 0.82);
            padding: 1rem 1.05rem;
            box-shadow: 0 10px 24px rgba(23, 32, 51, 0.04);
        }
        .asset-metric-label {
            color: var(--asset-muted);
            font-size: 0.82rem;
            font-weight: 700;
            letter-spacing: 0.10em;
            text-transform: uppercase;
            margin-bottom: 0.8rem;
        }
        .asset-metric-value {
            color: var(--asset-text);
            font-size: 1.48rem;
            line-height: 1.1;
            font-weight: 700;
            letter-spacing: -0.03em;
            margin-bottom: 0.75rem;
        }
        .asset-metric-note {
            color: var(--asset-muted);
            font-size: 0.88rem;
            line-height: 1.5;
        }
        .asset-metric-note.unavailable {
            color: #9c3131;
        }
        .asset-table-wrap {
            margin-top: 0.75rem;
            border: 1px solid var(--asset-border);
            border-radius: 20px;
            overflow: auto;
            background: rgba(255, 255, 255, 0.92);
            box-shadow: 0 10px 24px rgba(23, 32, 51, 0.04);
        }
        .asset-table {
            width: 100%;
            border-collapse: collapse;
            min-width: 620px;
        }
        .asset-table thead th {
            background: #f2efe8;
            color: var(--asset-text);
            font-size: 0.82rem;
            font-weight: 700;
            letter-spacing: 0.05em;
            text-transform: uppercase;
            text-align: left;
            padding: 0.9rem 1rem;
            border-bottom: 1px solid var(--asset-border);
            white-space: nowrap;
        }
        .asset-table tbody td {
            background: rgba(255, 255, 255, 0.98);
            color: var(--asset-text);
            font-size: 0.94rem;
            line-height: 1.45;
            padding: 0.92rem 1rem;
            border-bottom: 1px solid rgba(21, 33, 53, 0.06);
            vertical-align: top;
        }
        .asset-table tbody tr:last-child td {
            border-bottom: none;
        }
        .asset-table tbody tr:nth-child(even) td {
            background: #fcfbf8;
        }
        section.main [data-testid="stExpander"] {
            border: 1px solid var(--asset-border) !important;
            border-radius: 20px !important;
            background: rgba(255, 255, 255, 0.74) !important;
        }
        section.main [data-baseweb="input"] {
            border-radius: 16px !important;
            border-color: var(--asset-border) !important;
            background: rgba(255, 255, 255, 0.88) !important;
        }
        section.main [data-baseweb="input"] > div {
            background: transparent !important;
        }
        @media (max-width: 900px) {
            .asset-page-hero,
            .asset-summary-card {
                padding: 1.25rem 1.2rem;
            }
            .asset-summary-price {
                text-align: left;
            }
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


def _render_page_intro() -> None:
    st.markdown(
        """
        <div class="asset-page-hero">
            <div class="asset-page-kicker">ALFA / Equity Research</div>
            <div class="asset-page-title">Análise de Ativos</div>
            <div class="asset-page-subtitle">
                Consulte múltiplos indicadores fundamentalistas e de mercado em uma interface mais leve,
                limpa e focada na leitura. O objetivo aqui é reduzir ruído visual e deixar o diagnóstico do ativo
                mais elegante e direto.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _detail_value(snapshot: AssetSnapshot, field_name: str) -> str:
    selected = snapshot.details.loc[snapshot.details["Campo"] == field_name, "Valor"]
    if selected.empty:
        return ""
    value = selected.iloc[0]
    if value is None or pd.isna(value) or str(value).strip() == "N/A":
        return ""
    return str(value)


def _format_reference_note(reference_note: str) -> str:
    if not reference_note:
        return ""
    return escape(reference_note).replace("|", " &middot; ")


def _format_table_value(value: object) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return escape(str(value))


def _render_clean_table(dataframe: pd.DataFrame) -> None:
    if dataframe.empty:
        st.caption("Sem dados disponiveis.")
        return

    columns = "".join(f"<th>{escape(str(column))}</th>" for column in dataframe.columns)
    rows = []
    for _, row in dataframe.iterrows():
        cells = "".join(f"<td>{_format_table_value(value)}</td>" for value in row.tolist())
        rows.append(f"<tr>{cells}</tr>")

    st.markdown(
        f"""
        <div class="asset-table-wrap">
            <table class="asset-table">
                <thead>
                    <tr>{columns}</tr>
                </thead>
                <tbody>
                    {''.join(rows)}
                </tbody>
            </table>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_asset_summary(snapshot: AssetSnapshot) -> None:
    weekly_change = snapshot.metrics["weekly_price_change_pct"]
    if weekly_change.is_available:
        weekly_value = float(weekly_change.value)
        change_class = "positive" if weekly_value > 0 else "negative" if weekly_value < 0 else "neutral"
        change_label = f"Variacao semanal {weekly_change.formatted()}"
    else:
        change_class = "neutral"
        change_label = "Variacao semanal indisponivel"

    pills = [snapshot.ticker, _detail_value(snapshot, "Setor"), _detail_value(snapshot, "Indústria"), _detail_value(snapshot, "Bolsa"), _detail_value(snapshot, "País")]
    pills = [pill for pill in pills if pill]
    pill_markup = "".join(f'<div class="asset-pill">{escape(pill)}</div>' for pill in pills)

    st.markdown(
        f"""
        <div class="asset-summary-card">
            <div class="asset-summary-kicker">Snapshot</div>
            <div class="asset-summary-header">
                <div>
                    <div class="asset-summary-name">{escape(snapshot.long_name)}</div>
                    <div class="asset-summary-code">{escape(snapshot.ticker)} · moeda {escape(snapshot.currency)}</div>
                </div>
                <div>
                    <div class="asset-summary-price">{escape(snapshot.metrics["current_price"].formatted())}</div>
                    <div class="asset-summary-change {change_class}">{escape(change_label)}</div>
                </div>
            </div>
            <div class="asset-pill-row">{pill_markup}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_section_heading(title: str, description: str) -> None:
    st.markdown(
        f"""
        <div class="asset-section-heading">
            <div class="asset-section-title">{escape(title)}</div>
            <div class="asset-section-copy">{escape(description)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_metric_card(column, metric: MetricValue, reference_note: str) -> None:
    with column:
        note_markup = f'<div class="asset-metric-note">{_format_reference_note(reference_note)}</div>' if reference_note else ""
        unavailable_markup = (
            f'<div class="asset-metric-note unavailable">{escape(metric.unavailable_reason)}</div>'
            if not metric.is_available
            else ""
        )
        st.markdown(
            f"""
            <div class="asset-metric-card">
                <div class="asset-metric-label">{escape(metric.label)}</div>
                <div class="asset-metric-value">{escape(metric.formatted())}</div>
                {unavailable_markup}
                {note_markup}
            </div>
            """,
            unsafe_allow_html=True,
        )


def _render_section(snapshot: AssetSnapshot, title: str, description: str, metric_keys: list[str]) -> None:
    _render_section_heading(title, description)
    columns = st.columns(len(metric_keys))
    for column, metric_key in zip(columns, metric_keys):
        _render_metric_card(column, snapshot.metrics[metric_key], _reference_note(snapshot, metric_key))


def _render_details(snapshot: AssetSnapshot) -> None:
    with st.expander("Detalhes completos retornados pelo Yahoo Finance"):
        _render_clean_table(snapshot.details)


def _render_metric_reference(snapshot: AssetSnapshot) -> None:
    _render_section_heading(
        "Referencia dos Indicadores",
        "Nem toda metrica tem serie historica consistente. Aqui ficam claras as datas-base e a disponibilidade por indicador.",
    )
    st.caption("Indicadores sem histórico suficiente ficam apenas com o último valor disponível e a respectiva data-base.")

    reference_df = snapshot.metric_reference.copy()
    without_history = reference_df[reference_df["Histórico disponível"] == "Não"].drop(columns=["key"])
    if without_history.empty:
        st.caption("Todos os indicadores exibidos acima possuem alguma série histórica utilizável no Yahoo Finance.")
    else:
        _render_clean_table(without_history)

    with st.expander("Mapa completo de disponibilidade"):
        _render_clean_table(reference_df.drop(columns=["key"]))


def render_asset_analysis_page(default_ticker: str = "") -> None:
    _render_asset_analysis_styles()
    _render_page_intro()

    if "asset_analysis_selected_ticker" not in st.session_state:
        st.session_state["asset_analysis_selected_ticker"] = default_ticker.strip()

    shell = st.container(border=True)
    with shell:
        st.markdown('<div class="asset-analysis-search-shell"></div>', unsafe_allow_html=True)
        st.markdown('<div class="asset-panel-kicker">Consulta</div>', unsafe_allow_html=True)
        st.markdown('<div class="asset-panel-title">Pesquisar ticker</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="asset-panel-copy">Busque um ativo para abrir o snapshot consolidado, com foco em leitura limpa e navegação objetiva.</div>',
            unsafe_allow_html=True,
        )
        with st.form("asset-analysis-form", clear_on_submit=False):
            input_col, action_col = st.columns([4.2, 1.1], gap="medium")
            with input_col:
                ticker = st.text_input(
                    "Ticker",
                    value=st.session_state["asset_analysis_selected_ticker"],
                    placeholder="Ex.: AAPL, MSFT, PETR4.SA, VALE3.SA",
                    key="asset-analysis-input",
                )
            with action_col:
                st.caption("")
                submitted = st.form_submit_button("Analisar", use_container_width=True, type="primary")

    if submitted and ticker.strip():
        st.session_state["asset_analysis_selected_ticker"] = ticker.strip()

    analyzed_ticker = st.session_state["asset_analysis_selected_ticker"].strip()

    if not submitted and not analyzed_ticker:
        shell.markdown(
            '<div class="asset-empty-state">Informe um ticker e clique em <strong>Analisar</strong> para abrir a leitura do ativo.</div>',
            unsafe_allow_html=True,
        )
        return

    if submitted and not ticker.strip():
        shell.warning("Digite um ticker válido para iniciar a consulta.")
        return

    try:
        with st.spinner("Consultando Yahoo Finance..."):
            snapshot = fetch_asset_snapshot(analyzed_ticker)
    except MarketDataError as exc:
        shell.error(str(exc))
        return
    except Exception as exc:
        shell.error(f"Erro inesperado ao montar a análise do ativo: {exc}")
        return

    _render_asset_summary(snapshot)

    _render_section(
        snapshot,
        "Preço & Mercado",
        "Indicadores de precificacao e escala do ativo para leitura rapida da situacao atual.",
        ["current_price", "weekly_price_change_pct", "market_cap", "enterprise_value"],
    )
    _render_section(
        snapshot,
        "Valuation",
        "Multiplicadores e retorno sobre capital para enquadrar qualidade relativa e nivel de precificacao.",
        ["roic", "pe_ratio", "dividend_yield", "ev_to_ebitda"],
    )
    _render_section(
        snapshot,
        "Qualidade & Risco",
        "Margens, alavancagem e sensibilidade do papel para completar a leitura fundamentalista.",
        ["ebitda_margin", "financial_leverage", "beta"],
    )

    _render_metric_reference(snapshot)

    st.caption("Quando um campo aparece como N/A, o dado não foi disponibilizado pelo Yahoo Finance para este ticker.")
    _render_details(snapshot)
