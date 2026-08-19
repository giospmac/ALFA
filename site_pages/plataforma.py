"""Plataforma quant — hub das ferramentas de análise do ALFA.

As telas em si continuam vivendo em `ui/`; aqui ficam apenas a casca
(cabeçalho, sub-navegação e área útil) e o registro de ferramentas.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import streamlit as st

from theme import components as c

SUBNAV_KEY = "alfa_tool_pills"
STATE_KEY = "tool"


@dataclass(frozen=True)
class Tool:
    key: str
    label: str
    grupo: str
    resumo: str
    render: Callable[[], None]


def _tools() -> tuple[Tool, ...]:
    # importado aqui dentro para o app subir rápido e só carregar o que usar
    from ui.apt import render_apt_page
    from ui.asset_analysis import render_asset_analysis_page
    from ui.black_litterman import render_black_litterman_page
    from ui.charts import render_charts_page
    from ui.home import render_home_page
    from ui.markowitz import render_markowitz_page
    from ui.quant_projections import render_quant_projections_page
    from ui.risk_analysis import render_risk_analysis_page
    from ui.stock_comparison import render_stock_comparison_page
    from ui.stress_scenarios import render_stress_scenarios_page

    return (
        Tool("carteira", "Carteira", "Fundo", "Posições, pesos e histórico consolidado.", render_home_page),
        Tool("historico", "Histórico", "Fundo", "Retorno vs. benchmark, drawdown e Monte Carlo.", render_charts_page),
        Tool("risco", "Risco", "Fundo", "VaR, CVaR, CAPM, correlação e índices ajustados.", render_risk_analysis_page),
        Tool("stress", "Stress Test", "Fundo", "Cenários históricos e choques customizados.", render_stress_scenarios_page),
        Tool("markowitz", "Markowitz", "Alocação", "Fronteira eficiente e carteiras ótimas.", render_markowitz_page),
        Tool("bl", "Black-Litterman", "Alocação", "Retornos de equilíbrio combinados com suas visões.", render_black_litterman_page),
        Tool("apt", "APT", "Alocação", "Regressão multifatorial contra fatores macro.", render_apt_page),
        Tool("ativos", "Análise de Ativos", "Ativos", "Fundamentos, preço e risco de um ticker.", render_asset_analysis_page),
        Tool("comparador", "Comparador", "Ativos", "Até cinco ações lado a lado.", render_stock_comparison_page),
        Tool("quant", "Projeções Quant", "Ativos", "Projeção de preços por séries temporais.", render_quant_projections_page),
        # `ui/operations.py` (registro de compras/vendas/aportes) fica de fora
        # por ser tela de gestão interna. Para publicá-la, importe
        # `render_operations_page` e acrescente uma Tool aqui.
    )


def _current(tools: tuple[Tool, ...]) -> Tool:
    key = st.session_state.get(STATE_KEY, tools[0].key)
    return next((tool for tool in tools if tool.key == key), tools[0])


def _on_subnav_change(labels: dict[str, str]) -> None:
    label = st.session_state.get(SUBNAV_KEY)
    if label is None:  # clique na pílula já ativa desmarca — mantemos a seleção
        current_key = st.session_state.get(STATE_KEY, "")
        st.session_state[SUBNAV_KEY] = labels.get(current_key, next(iter(labels.values())))
        return
    st.session_state[STATE_KEY] = next(k for k, v in labels.items() if v == label)


def _render_header(tool: Tool) -> None:
    c.render(
        c.section(
            c.container(
                c.reveal(c.eyebrow("Plataforma quant"), step=1)
                + c.reveal("<h1>Ferramentas de análise</h1>", step=2)
                + c.reveal(
                    c.lead(
                        "Os modelos que a diretoria de Risco &amp; Quant usa para gerir o fundo simulado — "
                        "rodando ao vivo, com dados de mercado do Brasil e dos EUA."
                    ),
                    step=3,
                )
            ),
            variant="dark",
            waves="tl",
            extra="alfa-section--tight",
        )
    )


def render(*, goto=None) -> None:  # noqa: ARG001 — assinatura comum às páginas
    tools = _tools()
    if STATE_KEY not in st.session_state:
        st.session_state[STATE_KEY] = tools[0].key

    labels = {tool.key: tool.label for tool in tools}
    current = _current(tools)

    _render_header(current)

    with st.container(key="alfa_subnav"):
        st.session_state.setdefault(SUBNAV_KEY, labels[current.key])
        st.pills(
            "Ferramenta",
            list(labels.values()),
            key=SUBNAV_KEY,
            label_visibility="collapsed",
            on_change=_on_subnav_change,
            args=(labels,),
        )

    current = _current(tools)
    with st.container(key="alfa_page"):
        current.render()
