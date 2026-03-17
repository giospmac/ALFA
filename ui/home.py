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
    # Formata no padrão US (1,000.00) e inverte os separadores para o padrão BR (1.000,00)
    us_format = f"{value:,.2f}"
    br_format = us_format.replace(",", "X").replace(".", ",").replace("X", ".")
    return f"R$ {br_format}"

def _format_br_number(value: float, is_currency: bool = False) -> str:
    if pd.isna(value):
        return ""
    us_format = f"{value:,.2f}"
    br_format = us_format.replace(",", "X").replace(".", ",").replace("X", ".")
    return f"R$ {br_format}" if is_currency else br_format


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
    st.title("Portfólio")
    st.caption("Monte a carteira com acoes e titulos publicos, gere o historico consolidado e compare com benchmark.")

    if st.session_state.get("portfolio_notice"):
        st.success(st.session_state["portfolio_notice"])
        st.session_state["portfolio_notice"] = ""

    total_pl = st.number_input(
        "Patrimonio Liquido (R$)",
        min_value=0.0,
        value=float(st.session_state["portfolio_total_pl"]),
        step=100000.0,
        format="%.2f",
    )

    _sync_total_pl(total_pl)

    st.markdown('<div class="alfa-section-title" style="margin-top: 1rem;">Ações Rápidas</div>', unsafe_allow_html=True)
    toolbar_col_1, toolbar_col_2, toolbar_col_3 = st.columns([2, 2, 6])
    with toolbar_col_1:
        if st.button("🔄 Atualizar Histórico", type="primary", use_container_width=True):
            _refresh_history()
    with toolbar_col_2:
        if st.button("🧮 Recalcular Quantidades", use_container_width=True):
            st.session_state["portfolio_df"] = recalculate_portfolio(st.session_state["portfolio_df"], total_pl)
            _save_portfolio()
            st.session_state["portfolio_notice"] = "Quantidades recalculadas com base no PL atual."
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    add_tab_equity, add_tab_treasury = st.tabs(["Bolsa de Valores", "Titulos Publicos"])

    with add_tab_equity:
        with st.form("equity-form", clear_on_submit=True):
            equity_col_1, equity_col_2 = st.columns([2, 1])
            ticker_input = equity_col_1.text_input("Ticker", placeholder="Ex.: PETR4, PETR4.SA, AAPL, MSFT")
            target_weight = equity_col_2.number_input("Peso (%)", min_value=0.0, max_value=100.0, step=0.5, format="%.2f")
            add_equity = st.form_submit_button("➕ Adicionar ativo", use_container_width=True)

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
        selected_type = st.selectbox("Tipo de titulo", TREASURY_TYPES, key="treasury-type")
        available_years = treasury_year_options(treasury_df, selected_type)
        default_year = available_years[0] if available_years else datetime.now().year + 1

        with st.form("treasury-form", clear_on_submit=True):
            treasury_col_1, treasury_col_2 = st.columns([2, 1])
            selected_year = treasury_col_1.selectbox("Ano de vencimento", available_years or [default_year])
            treasury_weight = treasury_col_2.number_input("Peso (%)", min_value=0.0, max_value=100.0, step=0.5, format="%.2f", key="treasury-weight")
            add_treasury = st.form_submit_button("➕ Adicionar titulo", use_container_width=True)

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

    st.subheader("Carteira atual")
    if portfolio_df.empty:
        st.info("Adicione pelo menos um ativo para montar o portfolio.")
    else:
        # Formatar colunas para o padrão BR antes de exibir
        display_df = _portfolio_table_view(portfolio_df, total_pl)
        if not display_df.empty:
            for col in ["Preco/Titulo", "Valor Real (R$)"]:
                if col in display_df.columns:
                    display_df[col] = display_df[col].apply(lambda x: _format_br_number(x, True))
            for col in ["Taxa (%)", "Peso Desejado (%)", "Peso Real (%)", "Quantidade"]:
                if col in display_df.columns:
                    display_df[col] = display_df[col].apply(lambda x: _format_br_number(x, False))

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
        )

        remove_col_1, remove_col_2 = st.columns([3, 1])
        selected_ticker = remove_col_1.selectbox("Remover ativo", portfolio_df["ticker"].astype(str).tolist())
        with remove_col_2:
            st.markdown('<div class="align-remove-btn" style="display: none;"></div>', unsafe_allow_html=True)
            if st.button("Remover selecionado", use_container_width=True):
                st.session_state["portfolio_df"] = remove_position(portfolio_df, selected_ticker, total_pl)
                _save_portfolio()
                _refresh_history(show_success=False)
                st.session_state["portfolio_notice"] = f"{selected_ticker} removido do portfolio."
                st.rerun()

    st.subheader("Evolucao normalizada")
    normalized_df = normalized_history_with_portfolio(portfolio_df, historical_df)
    if normalized_df.empty:
        st.info("Atualize o historico para visualizar a evolucao consolidada da carteira.")
    else:
        st.line_chart(normalized_df, use_container_width=True)

    with st.expander("Historico bruto consolidado"):
        if historical_df.empty:
            st.write("Nenhum historico disponivel.")
        else:
            st.dataframe(historical_df.tail(120), use_container_width=True)
