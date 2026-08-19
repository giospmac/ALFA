"""Página Membros — grade por diretoria, alimentada por content/membros.json."""

from __future__ import annotations

import streamlit as st

from site_pages._shared import load_content, photo_uri
from theme import components as c

DESCRICOES = {
    "Presidência": "Direção geral do núcleo e das duas frentes, ALFA Asset e ALFA Núcleo.",
    "Equity Research": "Análise setorial, valuation e construção das teses de investimento.",
    "Risco & Quant": "Controle de risco, performance vs. benchmark, automação e modelos de alocação.",
    "Mercado": "Palestras, visitas institucionais, competições e oportunidades de carreira.",
    "Pessoas": "Processo seletivo, período trainee e programa de capacitação.",
}


def _render_header() -> None:
    c.render(
        c.section(
            c.container(
                c.reveal(c.eyebrow("Quem faz o ALFA"), step=1)
                + c.reveal("<h1>Membros</h1>", step=2)
                + c.reveal(
                    c.lead(
                        "Mais de 30 alunos organizados em quatro diretorias, sob mentoria acadêmica do "
                        "Departamento de Economia da PUC-Rio."
                    ),
                    step=3,
                )
            ),
            variant="dark",
            waves="tl",
            extra="alfa-section--tight",
        )
    )


def _diretoria_block(nome: str, pessoas: list[dict]) -> str:
    """Um grupo de membros: cabeçalho da diretoria + grade de pessoas."""
    if not pessoas:
        return ""
    cards = [
        c.member(
            nome=pessoa.get("nome", ""),
            cargo=pessoa.get("cargo", ""),
            foto_uri=photo_uri("membros", pessoa.get("foto", "")),
            linkedin=pessoa.get("linkedin", ""),
            step=(index % 6) + 1,
        )
        for index, pessoa in enumerate(pessoas)
    ]
    contagem = f"{len(pessoas)} {'membro' if len(pessoas) == 1 else 'membros'}"
    return (
        '<div style="margin-bottom:clamp(44px,6vw,72px)">'
        + c.section_head(kicker=contagem, title=c.esc(nome), subtitle=DESCRICOES.get(nome, ""))
        + c.grid(cards, cols=4)
        + "</div>"
    )


def _render_cta(on_process) -> None:
    with st.container(key="alfaband_dark_membroscta"):
        c.render(
            '<div class="alfa-center">'
            + c.section_head(
                title="Quer aparecer aqui?",
                subtitle="O processo seletivo abre a cada semestre.",
                center=True,
            )
            + "</div>"
        )
        with c.cta_row("membros"):
            if st.button("Conhecer o processo", key="membros_cta", type="primary"):
                on_process()


def render(*, goto) -> None:
    _render_header()

    content = load_content("membros")
    membros = content.get("membros", [])
    ordem = content.get("diretorias") or sorted({m.get("diretoria", "") for m in membros})

    if not membros:
        c.render(
            c.section(
                c.container(
                    '<p class="alfa-lead alfa-center">A lista de membros ainda não foi preenchida. '
                    "Edite <code>content/membros.json</code> para publicá-la.</p>"
                ),
                variant="light",
            )
        )
    else:
        blocos = "".join(
            _diretoria_block(diretoria, [m for m in membros if m.get("diretoria") == diretoria])
            for diretoria in ordem
        )
        c.render(c.section(c.container(blocos), variant="light"))

    _render_cta(lambda: goto("processo"))
