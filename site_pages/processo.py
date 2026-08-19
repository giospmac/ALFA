"""Página Processo Seletivo — jornada do membro e calendário do semestre."""

from __future__ import annotations

import streamlit as st

from site_pages._shared import load_content
from theme import components as c

JORNADA = [
    {"titulo": "Inscrição", "descricao": "Formulário, avaliação técnica e comportamental, e entrevistas."},
    {"titulo": "Integração", "descricao": "Onboarding institucional, apresentação das diretorias e projetos, e acesso aos materiais de capacitação."},
    {"titulo": "Período Trainee", "descricao": "12 aulas obrigatórias de fundamentos de mercado, equity, valuation e risco. Presença mínima de 80%."},
    {"titulo": "Extensão", "descricao": "Aulas avançadas em duas trilhas — Valuation e Finanças Quantitativas & Risco — abertas de forma opcional."},
    {"titulo": "Desafios Internos", "descricao": "Dois desafios em conjunto com os membros, focados em gestão de ativos e análise de ações."},
    {"titulo": "Fundo de Investimento", "descricao": "Trainees efetivados atuam no fundo simulado: screening, análises, modelos, teses, alocação e risco."},
    {"titulo": "Networking", "descricao": "Aulas com profissionais, visitas a bancos e gestoras, mentorias, imersões e competições acadêmicas."},
    {"titulo": "Continuidade & Legado", "descricao": "Trilhas avançadas, projetos mais complexos e posições de liderança nos semestres seguintes."},
]


def render(*, goto) -> None:
    c.render(
        c.section(
            c.container(
                c.reveal(c.eyebrow("Quero participar, e agora?"), step=1)
                + c.reveal("<h1>Jornada do Membro</h1>", step=2)
                + c.reveal(
                    c.lead(
                        "Da inscrição no processo seletivo à liderança do núcleo: as oito etapas da "
                        "trajetória de um membro do ALFA."
                    ),
                    step=3,
                )
            ),
            variant="dark",
            waves="tl",
            extra="alfa-section--tight",
        )
    )

    c.render(
        c.section(
            c.container(
                c.section_head(kicker="Etapa a etapa", title="Como funciona a trajetória", center=True)
                + f'<div style="max-width:760px;margin:0 auto">{c.timeline(JORNADA)}</div>'
            ),
            variant="light",
        )
    )

    eventos = load_content("eventos")
    itens = eventos.get("eventos", [])
    if itens:
        c.render(
            c.section(
                c.container(
                    c.section_head(
                        kicker=f"Calendário {eventos.get('semestre', '')}".strip(),
                        title="Capacitações &amp; visitas do semestre",
                        center=True,
                    )
                    + f'<div class="alfa-card alfa-reveal" style="max-width:760px;margin:0 auto">{c.agenda(itens)}</div>'
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
                    title="Ficou interessado?",
                    subtitle="As inscrições abrem a cada início de semestre. Acompanhe nossos canais para "
                    "não perder o prazo.",
                    center=True,
                )
                + c.ctas(
                    c.button("Seguir @alfapucrio", "https://www.instagram.com/alfapucrio", new_tab=True),
                    c.button("Falar com o ALFA", "mailto:alfapucrio@gmail.com", variant="outline"),
                )
                + "</div>"
            ),
            variant="light",
        )
    )
