from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from core.portfolio_repository import PortfolioRepository
from services.portfolio_analytics import (
    PortfolioAnalyticsError,
    append_position,
    build_historical_dataset,
    create_equity_position,
    create_treasury_position,
    empty_portfolio_frame,
    ensure_portfolio_schema,
    load_treasury_data,
    normalized_history_with_portfolio,
    portfolio_overview,
    recalculate_portfolio,
    remove_position,
    treasury_year_options,
    TREASURY_TYPES,
)


def _get_repository() -> PortfolioRepository:
    return PortfolioRepository(base_path=Path(__file__).resolve().parents[1])


def _bootstrap_state() -> None:
    if "portfolio_df" in st.session_state and "historical_df" in st.session_state:
        return

    snapshot = _get_repository().load_snapshot()
    portfolio_df = ensure_portfolio_schema(snapshot.portfolio)
    historical_df = snapshot.historical.copy()
    total_pl = float(pd.to_numeric(portfolio_df["valor_real"], errors="coerce").fillna(0).sum())

    st.session_state["portfolio_df"] = portfolio_df
    st.session_state["historical_df"] = historical_df
    st.session_state["portfolio_total_pl"] = total_pl if total_pl > 0 else 100000000.0
    st.session_state["invalid_tickers"] = []
    st.session_state["portfolio_notice"] = ""


def _save_portfolio() -> None:
    repo = _get_repository()
    repo.save_portfolio_data(st.session_state["portfolio_df"])


def _save_historical() -> None:
    repo = _get_repository()
    repo.save_historical_data(st.session_state["historical_df"])


def _refresh_history(show_success: bool = True) -> None:
    with st.spinner("Atualizando historico, benchmarks e titulos publicos..."):
        result = build_historical_dataset(st.session_state["portfolio_df"])
    st.session_state["historical_df"] = result.historical
    st.session_state["invalid_tickers"] = result.invalid_tickers
    _save_historical()
    if show_success:
        st.session_state["portfolio_notice"] = "Historico atualizado com sucesso."


def _sync_total_pl(total_pl: float) -> None:
    current_total_pl = float(st.session_state["portfolio_total_pl"])
    if abs(total_pl - current_total_pl) < 0.01:
        return

    st.session_state["portfolio_total_pl"] = total_pl
    st.session_state["portfolio_df"] = recalculate_portfolio(st.session_state["portfolio_df"], total_pl)
    _save_portfolio()


def _format_currency(value: float) -> str:
    return f"R$ {value:,.2f}"


def _render_home_styles() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(180deg, #f4f5f0 0%, #eef1f7 100%);
        }
        .alfa-home-hero {
            background: linear-gradient(135deg, #102170 0%, #17318f 58%, #4979f6 100%);
            border-radius: 24px;
            padding: 1.6rem 1.7rem;
            color: #f4f5f0;
            box-shadow: 0 20px 44px rgba(16, 33, 112, 0.22);
            margin-bottom: 1rem;
        }
        .alfa-home-kicker {
            color: #97bdff;
            font-size: 0.82rem;
            font-weight: 700;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            margin-bottom: 0.55rem;
        }
        .alfa-home-title {
            color: #f4f5f0;
            font-size: 2.1rem;
            font-weight: 700;
            line-height: 1.05;
            margin-bottom: 0.45rem;
        }
        .alfa-home-subtitle {
            color: rgba(244, 245, 240, 0.88);
            font-size: 1rem;
            max-width: 56rem;
            line-height: 1.5;
        }
        .alfa-section-title {
            color: #102170;
            font-size: 1.02rem;
            font-weight: 700;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            margin: 0.3rem 0 0.85rem 0.1rem;
        }
        .alfa-kpi-card {
            background: #ffffff;
            border: 1px solid rgba(73, 121, 246, 0.14);
            border-radius: 20px;
            padding: 1rem 1.05rem;
            box-shadow: 0 12px 30px rgba(16, 33, 112, 0.06);
            min-height: 118px;
        }
        .alfa-kpi-label {
            color: #6e7ba8;
            font-size: 0.84rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-weight: 700;
            margin-bottom: 0.75rem;
        }
        .alfa-kpi-value {
            color: #102170;
            font-size: 1.55rem;
            font-weight: 700;
            line-height: 1.1;
            margin-bottom: 0.45rem;
        }
        .alfa-kpi-note {
            color: #4979f6;
            font-size: 0.9rem;
            font-weight: 600;
        }
        .alfa-surface {
            background: rgba(255, 255, 255, 0.82);
            border: 1px solid rgba(73, 121, 246, 0.12);
            border-radius: 22px;
            padding: 1.05rem;
            box-shadow: 0 14px 34px rgba(16, 33, 112, 0.05);
        }
        [data-testid="stNumberInput"] label,
        [data-testid="stTextInput"] label,
        [data-testid="stSelectbox"] label {
            color: #102170;
            font-weight: 600;
        }
        [data-testid="stTextInput"] input,
        [data-testid="stNumberInput"] input,
        [data-testid="stSelectbox"] div[data-baseweb="select"] > div,
        [data-testid="stSelectbox"] div[data-baseweb="select"] input {
            background: #ffffff;
            border-color: rgba(73, 121, 246, 0.2);
        }
        div[data-testid="stButton"] > button,
        div[data-testid="stFormSubmitButton"] > button {
            border-radius: 14px;
            border: 1px solid rgba(73, 121, 246, 0.18);
            background: #ffffff;
            color: #102170;
            min-height: 2.8rem;
            font-weight: 700;
            box-shadow: none;
        }
        div[data-testid="stButton"] > button[kind="primary"],
        div[data-testid="stFormSubmitButton"] > button[kind="primary"] {
            background: #4979f6;
            color: #f4f5f0;
            border-color: #4979f6;
        }
        div[data-testid="stTabs"] [data-baseweb="tab-list"] {
            gap: 0.5rem;
            margin-bottom: 0.6rem;
        }
        div[data-testid="stTabs"] button[role="tab"] {
            border-radius: 999px;
            padding: 0.45rem 1rem;
            background: rgba(151, 189, 255, 0.14);
            color: #102170;
            border: 1px solid rgba(73, 121, 246, 0.14);
        }
        div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
            background: #102170;
            color: #f4f5f0;
        }
        [data-testid="stDataFrame"],
        [data-testid="stExpander"] details,
        [data-testid="stAlert"] {
            background: rgba(255, 255, 255, 0.86);
            border-radius: 18px;
            border: 1px solid rgba(73, 121, 246, 0.12);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_section_title(title: str) -> None:
    st.markdown(f'<div class="alfa-section-title">{title}</div>', unsafe_allow_html=True)


def _render_overview_card(column, label: str, value: str, note: str) -> None:
    with column:
        st.markdown(
            (
                '<div class="alfa-kpi-card">'
                f'<div class="alfa-kpi-label">{label}</div>'
                f'<div class="alfa-kpi-value">{value}</div>'
                f'<div class="alfa-kpi-note">{note}</div>'
                "</div>"
            ),
            unsafe_allow_html=True,
        )


def _portfolio_table_view(portfolio_df: pd.DataFrame, total_pl: float) -> pd.DataFrame:
    if portfolio_df.empty:
        return pd.DataFrame()

    view_df = ensure_portfolio_schema(portfolio_df).copy()
    view_df = view_df.sort_values("porcentagem_real", ascending=False)
    view_df = view_df.rename(
        columns={
            "ticker": "Ticker",
            "nome": "Nome",
            "tipo_ativo": "Tipo",
            "preco": "Preco/Titulo",
            "taxa": "Taxa (%)",
            "porcentagem_desejada": "Peso Desejado (%)",
            "porcentagem_real": "Peso Real (%)",
            "qtd_acoes": "Quantidade",
            "valor_real": "Valor Real (R$)",
            "data_fechamento": "Data Base",
            "data_vencimento": "Vencimento",
        }
    )

    cash_remaining = max(total_pl - pd.to_numeric(portfolio_df["valor_real"], errors="coerce").fillna(0).sum(), 0.0)
    cash_row = pd.DataFrame(
        [
            {
                "Ticker": "Valor Restante",
                "Nome": "",
                "Tipo": "caixa",
                "Preco/Titulo": "",
                "Taxa (%)": "",
                "Peso Desejado (%)": "",
                "Peso Real (%)": (cash_remaining / total_pl) * 100 if total_pl > 0 else 0.0,
                "Quantidade": "",
                "Valor Real (R$)": cash_remaining,
                "Data Base": "",
                "Vencimento": "",
            }
        ]
    )
    view_df = pd.concat([view_df, cash_row], ignore_index=True)

    for column in ["Preco/Titulo", "Taxa (%)", "Peso Desejado (%)", "Peso Real (%)", "Quantidade", "Valor Real (R$)"]:
        if column in view_df.columns:
            view_df[column] = pd.to_numeric(view_df[column], errors="coerce")

    return view_df


def render_home_page() -> None:
    _bootstrap_state()
    _render_home_styles()

    st.markdown(
        """
        <div class="alfa-home-hero">
            <div class="alfa-home-kicker">Portfolio workspace</div>
            <div class="alfa-home-title">Construa, rebalanceie e acompanhe sua carteira</div>
            <div class="alfa-home-subtitle">
                Ajuste pesos, combine bolsa e títulos públicos, atualize o histórico consolidado e acompanhe a evolução
                da alocação com uma interface mais limpa e focada.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    _render_section_title("Configuração")

    if st.session_state.get("portfolio_notice"):
        st.success(st.session_state["portfolio_notice"])
        st.session_state["portfolio_notice"] = ""

    config_col_1, config_col_2 = st.columns([1.3, 1], gap="large")
    with config_col_1:
        st.markdown('<div class="alfa-surface">', unsafe_allow_html=True)
        total_pl = st.number_input(
            "Patrimonio Liquido (R$)",
            min_value=0.0,
            value=float(st.session_state["portfolio_total_pl"]),
            step=100000.0,
            format="%.2f",
        )
        st.markdown("</div>", unsafe_allow_html=True)
    with config_col_2:
        st.markdown('<div class="alfa-surface">', unsafe_allow_html=True)
        st.caption("Ações rápidas")
        controls_col_1, controls_col_2 = st.columns([1, 1])
        with controls_col_1:
            if st.button("Atualizar histórico", use_container_width=True, type="primary"):
                _refresh_history()
        with controls_col_2:
            if st.button("Recalcular lotes", use_container_width=True):
                st.session_state["portfolio_df"] = recalculate_portfolio(st.session_state["portfolio_df"], total_pl)
                _save_portfolio()
                st.session_state["portfolio_notice"] = "Quantidades recalculadas com base no PL atual."
                st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    _sync_total_pl(total_pl)

    add_tab_equity, add_tab_treasury = st.tabs(["Bolsa de Valores", "Titulos Publicos"])

    with add_tab_equity:
        st.markdown('<div class="alfa-surface">', unsafe_allow_html=True)
        with st.form("equity-form", clear_on_submit=True):
            equity_col_1, equity_col_2 = st.columns([2, 1])
            ticker_input = equity_col_1.text_input("Ticker", placeholder="Ex.: PETR4, PETR4.SA, AAPL, MSFT")
            target_weight = equity_col_2.number_input("Peso (%)", min_value=0.0, max_value=100.0, step=0.5, format="%.2f")
            add_equity = st.form_submit_button("Adicionar ativo", use_container_width=True, type="primary")
        st.markdown("</div>", unsafe_allow_html=True)

        if add_equity:
            try:
                position = create_equity_position(st.session_state["portfolio_df"], ticker_input, target_weight, total_pl)
                st.session_state["portfolio_df"] = append_position(st.session_state["portfolio_df"], position, total_pl)
                _save_portfolio()
                _refresh_history(show_success=False)
                st.session_state["portfolio_notice"] = f'{position["ticker"]} adicionado ao portfolio.'
                st.rerun()
            except PortfolioAnalyticsError as exc:
                st.error(str(exc))

    with add_tab_treasury:
        treasury_df = load_treasury_data()
        st.markdown('<div class="alfa-surface">', unsafe_allow_html=True)
        selected_type = st.selectbox("Tipo de titulo", TREASURY_TYPES, key="treasury-type")
        available_years = treasury_year_options(treasury_df, selected_type)
        default_year = available_years[0] if available_years else datetime.now().year + 1

        with st.form("treasury-form", clear_on_submit=True):
            treasury_col_1, treasury_col_2 = st.columns([2, 1])
            selected_year = treasury_col_1.selectbox("Ano de vencimento", available_years or [default_year])
            treasury_weight = treasury_col_2.number_input("Peso (%)", min_value=0.0, max_value=100.0, step=0.5, format="%.2f", key="treasury-weight")
            add_treasury = st.form_submit_button("Adicionar titulo", use_container_width=True, type="primary")
        st.markdown("</div>", unsafe_allow_html=True)

        if add_treasury:
            try:
                position = create_treasury_position(
                    st.session_state["portfolio_df"],
                    treasury_df,
                    selected_type,
                    int(selected_year),
                    treasury_weight,
                    total_pl,
                )
                st.session_state["portfolio_df"] = append_position(st.session_state["portfolio_df"], position, total_pl)
                _save_portfolio()
                _refresh_history(show_success=False)
                st.session_state["portfolio_notice"] = f'{position["ticker"]} adicionado ao portfolio.'
                st.rerun()
            except PortfolioAnalyticsError as exc:
                st.error(str(exc))

    if st.session_state["invalid_tickers"]:
        st.warning(
            "Tickers ignorados por falta de historico valido no Yahoo Finance: "
            + ", ".join(st.session_state["invalid_tickers"])
        )

    portfolio_df = st.session_state["portfolio_df"]
    historical_df = st.session_state["historical_df"]
    overview = portfolio_overview(portfolio_df, historical_df, total_pl)

    _render_section_title("Resumo")
    metric_col_1, metric_col_2, metric_col_3, metric_col_4 = st.columns(4)
    allocation_note = f'{(overview["invested_value"] / total_pl) * 100:.1f}% alocado' if total_pl > 0 else "0.0% alocado"
    cash_note = f'{(overview["cash_remaining"] / total_pl) * 100:.1f}% em caixa' if total_pl > 0 else "0.0% em caixa"
    _render_overview_card(metric_col_1, "Ativos", str(overview["asset_count"]), "posição monitorada")
    _render_overview_card(metric_col_2, "Investido", _format_currency(overview["invested_value"]), allocation_note)
    _render_overview_card(metric_col_3, "Caixa livre", _format_currency(overview["cash_remaining"]), cash_note)
    _render_overview_card(metric_col_4, "Último histórico", overview["latest_history_date"], "data base consolidada")

    _render_section_title("Carteira Atual")
    if portfolio_df.empty:
        st.info("Adicione pelo menos um ativo para montar o portfolio.")
    else:
        st.markdown('<div class="alfa-surface">', unsafe_allow_html=True)
        st.dataframe(
            _portfolio_table_view(portfolio_df, total_pl),
            use_container_width=True,
            hide_index=True,
            column_config={
                "Preco/Titulo": st.column_config.NumberColumn(format="R$ %.2f"),
                "Taxa (%)": st.column_config.NumberColumn(format="%.2f"),
                "Peso Desejado (%)": st.column_config.NumberColumn(format="%.2f"),
                "Peso Real (%)": st.column_config.NumberColumn(format="%.2f"),
                "Quantidade": st.column_config.NumberColumn(format="%.2f"),
                "Valor Real (R$)": st.column_config.NumberColumn(format="R$ %.2f"),
            },
        )

        remove_col_1, remove_col_2 = st.columns([3, 1])
        selected_ticker = remove_col_1.selectbox("Remover ativo", portfolio_df["ticker"].astype(str).tolist())
        if remove_col_2.button("Remover selecionado", use_container_width=True):
            st.session_state["portfolio_df"] = remove_position(portfolio_df, selected_ticker, total_pl)
            _save_portfolio()
            _refresh_history(show_success=False)
            st.session_state["portfolio_notice"] = f"{selected_ticker} removido do portfolio."
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    _render_section_title("Evolução Normalizada")
    normalized_df = normalized_history_with_portfolio(portfolio_df, historical_df)
    if normalized_df.empty:
        st.info("Atualize o historico para visualizar a evolucao consolidada da carteira.")
    else:
        st.markdown('<div class="alfa-surface">', unsafe_allow_html=True)
        st.line_chart(normalized_df, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with st.expander("Historico bruto consolidado"):
        if historical_df.empty:
            st.write("Nenhum historico disponivel.")
        else:
            st.dataframe(historical_df.tail(120), use_container_width=True)
