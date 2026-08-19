"""Página Atividades — visitas institucionais e competições."""

from __future__ import annotations

import streamlit as st

from site_pages._shared import load_content, photo_uri
from theme import components as c


def _visit_card(item: dict, step: int) -> str:
    confirmada = str(item.get("status", "")).lower() == "confirmada"
    badge = c.chip("Confirmada", tone="ok") if confirmada else c.chip("Agendando", tone="wait")
    foto = photo_uri("atividades", item.get("foto", ""))
    imagem = (
        f'<img src="{foto}" alt="{c.esc(item.get("nome", ""))}" '
        f'style="border-radius:10px;margin-bottom:14px;aspect-ratio:16/10;object-fit:cover;width:100%">'
        if foto
        else ""
    )
    return c.reveal(
        f'<div class="alfa-card">{imagem}<div style="margin-bottom:10px">{badge}</div>'
        f'<h3>{c.esc(item.get("nome", ""))}</h3>'
        f'<p>{c.esc(item.get("descricao", ""))}</p></div>',
        step=step,
    )


def _competition_card(item: dict, step: int) -> str:
    foto = photo_uri("atividades", item.get("foto", ""))
    imagem = (
        f'<img src="{foto}" alt="{c.esc(item.get("nome", ""))}" '
        f'style="border-radius:10px;margin-bottom:14px;aspect-ratio:16/10;object-fit:cover;width:100%">'
        if foto
        else ""
    )
    return c.reveal(
        f'<div class="alfa-card">{imagem}'
        f'<div style="margin-bottom:10px">{c.chip(item.get("area", ""), tone="info")}</div>'
        f'<h3>{c.esc(item.get("nome", ""))}</h3></div>',
        step=step,
    )


def render(*, goto) -> None:
    content = load_content("atividades")
    visitas = content.get("visitas", [])
    competicoes = content.get("competicoes", [])

    c.render(
        c.section(
            c.container(
                c.reveal(c.eyebrow("ALFA Núcleo · Mercado"), step=1)
                + c.reveal("<h1>Atividades &amp; Viagens</h1>", step=2)
                + c.reveal(
                    c.lead(
                        "Visitas institucionais, palestras com profissionais, imersões no mercado "
                        "financeiro e competições — a ponte entre a sala de aula e o buy-side."
                    ),
                    step=3,
                )
            ),
            variant="dark",
            waves="tl",
            extra="alfa-section--tight",
        )
    )

    if visitas:
        confirmadas = sum(1 for v in visitas if str(v.get("status", "")).lower() == "confirmada")
        c.render(
            c.section(
                c.container(
                    c.section_head(
                        kicker="Visitas institucionais",
                        title="Onde estivemos e para onde vamos",
                        subtitle=f"{confirmadas} visitas confirmadas e {len(visitas) - confirmadas} em "
                        "agendamento — bancos, gestoras e empresas do setor, incluindo imersões na "
                        "Faria Lima.",
                    )
                    + c.grid([_visit_card(v, (i % 6) + 1) for i, v in enumerate(visitas)], cols=3)
                ),
                variant="light",
            )
        )

    if competicoes:
        c.render(
            c.section(
                c.container(
                    c.section_head(
                        kicker="Competições & desafios",
                        title="Onde testamos o que aprendemos",
                        subtitle="Desafios de equity research, asset management, quant e wealth promovidos "
                        "por instituições do mercado.",
                    )
                    + c.grid([_competition_card(v, (i % 6) + 1) for i, v in enumerate(competicoes)], cols=3)
                ),
                variant="dark",
                waves="br",
            )
        )

    c.render(
        c.section(
            c.container(
                '<div class="alfa-center">'
                + c.section_head(
                    title="Acompanhe de perto",
                    subtitle="Fotos e bastidores das visitas e competições no nosso Instagram.",
                    center=True,
                )
                + c.ctas(c.button("@alfapucrio", "https://www.instagram.com/alfapucrio", new_tab=True))
                + "</div>"
            ),
            variant="light",
        )
    )
