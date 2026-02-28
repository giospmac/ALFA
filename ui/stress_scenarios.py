from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from core.portfolio_repository import PortfolioRepository
from services.portfolio_analytics import (
    StressScenarioCase,
    custom_stress_scenario,
    historical_stress_cases,
)


def _load_data():
    if "portfolio_df" in st.session_state and "historical_df" in st.session_state:
        total_pl = float(st.session_state.get("portfolio_total_pl", 100000000.0))
        return st.session_state["portfolio_df"], st.session_state["historical_df"], total_pl

    repo = PortfolioRepository(base_path=Path(__file__).resolve().parents[1])
    snapshot = repo.load_snapshot()
    invested_value = float(pd.to_numeric(snapshot.portfolio.get("valor_real"), errors="coerce").fillna(0).sum()) if not snapshot.portfolio.empty else 0.0
    total_pl = invested_value if invested_value > 0 else 100000000.0
    return snapshot.portfolio, snapshot.historical, total_pl


def _format_pct(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{value * 100:.2f}%"


def _format_currency(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"R$ {value:,.2f}"


def _summary_table(cases: list[StressScenarioCase]) -> pd.DataFrame:
    rows = []
    for case in cases:
        rows.append(
            {
                "Cenário": case.label,
                "Referência": case.reference,
                "Disponível": "Sim" if case.available else "Não",
                "Retorno esperado": None if case.expected_return is None else case.expected_return * 100,
                "Drawdown": None if case.drawdown is None else case.drawdown * 100,
                "VaR estressado": None if case.stressed_var is None else case.stressed_var * 100,
                "Impacto em valor": case.pnl_impact,
                "Observação": case.reason or "",
            }
        )
    return pd.DataFrame(rows)


def _plot_case(case: StressScenarioCase, title: str) -> None:
    if case.path.empty:
        st.info("Não há trajetória suficiente para exibir gráfico neste cenário.")
        return

    fig, ax = plt.subplots(figsize=(10, 4.8))
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")

    ax.plot(case.path.index, case.path.values, color="#4979f6", linewidth=2.4)
    ax.fill_between(case.path.index, case.path.values, case.path.min(), color="#97bdff", alpha=0.25)
    ax.grid(True, axis="y", linestyle="--", alpha=0.25)
    ax.grid(False, axis="x")
    ax.set_title(title)
    ax.set_ylabel("Base 100")
    ax.set_xlabel("Data")
    for spine in ax.spines.values():
        spine.set_color("#dddddd")

    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def _render_case_metrics(case: StressScenarioCase) -> None:
    metric_col_1, metric_col_2, metric_col_3, metric_col_4 = st.columns(4)
    metric_col_1.metric("Retorno esperado", _format_pct(case.expected_return))
    metric_col_2.metric("Drawdown", _format_pct(case.drawdown))
    metric_col_3.metric("VaR estressado", _format_pct(case.stressed_var))
    metric_col_4.metric("Impacto em valor", _format_currency(case.pnl_impact))
    st.caption(case.reference)


def _render_asset_impacts(case: StressScenarioCase) -> None:
    if case.asset_impacts.empty:
        st.info("Não foi possível decompor o impacto por ativo neste cenário.")
        return

    display_df = case.asset_impacts.copy()
    for column in ["Peso", "Retorno do ativo", "Impacto na carteira"]:
        display_df[column] = pd.to_numeric(display_df[column], errors="coerce") * 100

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Peso": st.column_config.NumberColumn(format="%.2f%%"),
            "Retorno do ativo": st.column_config.NumberColumn(format="%.2f%%"),
            "Impacto na carteira": st.column_config.NumberColumn(format="%.2f%%"),
            "Impacto em valor": st.column_config.NumberColumn(format="R$ %.2f"),
        },
    )


def _custom_scenario_controls() -> tuple[str, float]:
    scenario_type = st.selectbox(
        "Tipo de cenário",
        options=[
            ("market_drop", "Queda de mercado"),
            ("rate_hike", "Aumento de juros"),
            ("vol_spike", "Alta de volatilidade"),
        ],
        format_func=lambda item: item[1],
    )[0]

    if scenario_type == "market_drop":
        magnitude = st.select_slider("Choque", options=[10.0, 20.0, 30.0], value=10.0, format_func=lambda value: f"-{value:.0f}%")
    elif scenario_type == "rate_hike":
        magnitude = st.select_slider("Choque", options=[50.0, 100.0, 200.0, 300.0], value=100.0, format_func=lambda value: f"+{value:.0f} bps")
    else:
        magnitude = st.select_slider("Choque", options=[50.0, 100.0, 150.0], value=50.0, format_func=lambda value: f"+{value:.0f}%")

    return scenario_type, magnitude


def render_stress_scenarios_page() -> None:
    portfolio_df, historical_df, total_pl = _load_data()

    st.title("Stress Test & Cenários")
    st.caption("Stress histórico e cenários customizados para medir retorno esperado, drawdown e VaR estressado da carteira.")

    if portfolio_df.empty or historical_df.empty:
        st.info("Monte o portfólio e atualize o histórico na Home antes de abrir esta análise.")
        return

    historical_tab, custom_tab = st.tabs(["Stress histórico", "Cenários customizados"])

    with historical_tab:
        cases = historical_stress_cases(portfolio_df, historical_df, total_pl)
        if not cases:
            st.info("Não há dados suficientes para rodar os cenários históricos.")
        else:
            summary_df = _summary_table(cases)
            st.dataframe(
                summary_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Retorno esperado": st.column_config.NumberColumn(format="%.2f%%"),
                    "Drawdown": st.column_config.NumberColumn(format="%.2f%%"),
                    "VaR estressado": st.column_config.NumberColumn(format="%.2f%%"),
                    "Impacto em valor": st.column_config.NumberColumn(format="R$ %.2f"),
                },
            )

            available_cases = [case for case in cases if case.available]
            if not available_cases:
                st.info("A base histórica atual da carteira não cobre as janelas selecionadas.")
            else:
                selected_label = st.selectbox("Cenário histórico para detalhar", [case.label for case in available_cases], index=0)
                selected_case = next(case for case in available_cases if case.label == selected_label)
                _render_case_metrics(selected_case)
                _plot_case(selected_case, f"{selected_case.label} | trajetória da carteira")
                st.subheader("Impacto por ativo")
                _render_asset_impacts(selected_case)

    with custom_tab:
        control_col_1, control_col_2 = st.columns([1.4, 1.1])
        with control_col_1:
            scenario_type, magnitude = _custom_scenario_controls()
        with control_col_2:
            st.markdown("")
            st.markdown("")
            st.caption("Os choques são aplicados sobre a carteira atual, sem rebalanceamento.")

        custom_case = custom_stress_scenario(portfolio_df, historical_df, total_pl, scenario_type, magnitude)
        if custom_case is None:
            st.info("Não foi possível montar o cenário customizado com a base atual.")
            return

        _render_case_metrics(custom_case)
        _plot_case(custom_case, f"{custom_case.label} | impacto projetado")
        st.subheader("Impacto por ativo")
        _render_asset_impacts(custom_case)
