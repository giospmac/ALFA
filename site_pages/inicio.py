"""Página inicial — hero, quem somos, highlights, estrutura e chamadas."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from site_pages._shared import fund_snapshot, load_content, num, pct, short_ticker
from theme import components as c
from theme.styles import asset_data_uri

DIRETORIAS = [
    {
        "frente": "ALFA Asset",
        "nome": "Equity Research",
        "texto": "Transforma análise fundamentalista em teses com potencial de gerar alpha: "
        "análise setorial, valuation e tese de investimento.",
    },
    {
        "frente": "ALFA Asset",
        "nome": "Risco & Quant",
        "texto": "Controle de risco do portfólio, análise de performance vs. benchmark, "
        "automação e modelos de alocação.",
    },
    {
        "frente": "ALFA Núcleo",
        "nome": "Mercado",
        "texto": "Palestras com profissionais, visitas a instituições, preparação para "
        "competições e divulgação de oportunidades.",
    },
    {
        "frente": "ALFA Núcleo",
        "nome": "Pessoas",
        "texto": "Processo seletivo, período trainee e todo o programa de capacitação do núcleo.",
    },
]

TAGS = [
    "networking",
    "competições e desafios",
    "imersões no mercado",
    "mentorias",
    "treinamentos técnicos",
]


def _ticker_items(portfolio_df: pd.DataFrame, limit: int = 18) -> list[tuple[str, str, float]]:
    """Faixa rolante com a composição atual da carteira."""
    if portfolio_df.empty or "ticker" not in portfolio_df.columns:
        return []

    working = portfolio_df.copy()
    working["porcentagem_real"] = pd.to_numeric(working["porcentagem_real"], errors="coerce").fillna(0.0)
    working["preco"] = pd.to_numeric(working["preco"], errors="coerce")
    working = working[working["ticker"].astype(str).str.strip() != ""]
    working = working.sort_values("porcentagem_real", ascending=False).head(limit)

    items: list[tuple[str, str, float]] = []
    for _, row in working.iterrows():
        price = row["preco"]
        label = f"R$ {num(price)}" if pd.notna(price) else ""
        items.append((short_ticker(row["ticker"]), label, float(row["porcentagem_real"])))
    return items


def _render_hero(on_fund, on_process) -> None:
    with st.container(key="alfa_hero"):
        c.render(
            c.reveal(f'<img class="alfa-hero-logo" src="{asset_data_uri("logo-alfa-white.png")}" alt="ALFA">', step=1)
            + c.reveal('<h1 class="alfa-hero-title">ALFA</h1>', step=2)
            + c.reveal(
                '<p class="alfa-hero-sub">Laboratório de Finanças Aplicadas PUC-Rio · '
                "Departamento de Economia</p>",
                step=3,
            )
            + c.reveal(
                c.lead(
                    "Formamos talentos e lideranças para o mercado financeiro por meio de um fundo de "
                    "investimento long only simulado, com análises autorais."
                ),
                step=4,
            )
        )
        with c.cta_row("hero"):
            if st.button("Conheça o fundo", key="hero_cta_fundo", type="primary"):
                on_fund()
            if st.button("Processo seletivo", key="hero_cta_ps"):
                on_process()
        c.render(
            c.reveal('<div class="alfa-scrollcue"><span class="line"></span>role para explorar</div>', step=6)
        )


def _render_ticker(portfolio_df: pd.DataFrame) -> None:
    items = _ticker_items(portfolio_df)
    if not items:
        return
    cells = "".join(
        f'<span class="alfa-ticker__item"><b>{c.esc(symbol)}</b>{c.esc(price)} '
        f'<span class="up">{pct(weight)}</span></span>'
        for symbol, price, weight in items
    )
    c.render(
        f'<section class="alfa-section alfa-section--dark" style="padding:0">'
        f'<div class="alfa-ticker"><div class="alfa-ticker__track">{cells}{cells}</div></div>'
        f'<div class="alfa-container" style="padding-top:10px;padding-bottom:14px;text-align:center">'
        f'<span style="font-size:.7rem;letter-spacing:.16em;text-transform:uppercase;color:var(--on-dark-soft)">'
        f"Carteira simulada ALFA · preço de fechamento e peso alvo</span></div></section>"
    )


def _render_quem_somos() -> None:
    c.render(
        c.section(
            c.container(
                '<div class="alfa-split">'
                + c.reveal(
                    c.eyebrow("Quem somos")
                    + "<h2>Alfa é o retorno acima do benchmark, ajustado ao risco.</h2>",
                    step=1,
                )
                + c.reveal(
                    "<p>Fundado em 2024, o ALFA é o <strong>Núcleo de Finanças</strong> do Departamento de "
                    "Economia da PUC-Rio. Simulamos um <strong>fundo de investimento com foco em análise de "
                    "ações (equity, long-only)</strong>, produzindo análises autorais que embasam nossas "
                    "decisões de alocação no longo prazo.</p>"
                    "<p>Ao longo do semestre promovemos capacitações técnicas, palestras com profissionais do "
                    "mercado e projetos práticos. Nossas análises seguem a filosofia do <strong>Value "
                    "Investing</strong>: empresas de qualidade, com vantagens competitivas robustas, "
                    "negociadas abaixo do valor intrínseco.</p>"
                    f'<div style="margin-top:18px">{c.tags(TAGS)}</div>',
                    step=2,
                )
                + "</div>"
            ),
            variant="light",
        )
    )


def _render_highlights() -> None:
    content = load_content("highlights")
    itens = content.get("itens", [])
    mentor = content.get("mentor", {})
    if not itens:
        return

    stats = [
        c.stat(item.get("valor", ""), item.get("label", ""), step=index + 1)
        for index, item in enumerate(itens)
    ]
    mentor_markup = ""
    if mentor.get("nome"):
        mentor_markup = (
            f'<p class="alfa-center" style="margin-top:34px">'
            f'{c.esc(mentor.get("cargo", "Mentor Acadêmico"))}: '
            f'<strong style="color:var(--on-dark)">{c.esc(mentor["nome"])}</strong></p>'
        )

    c.render(
        c.section(
            c.container(
                c.section_head(kicker="Highlights", title="Os últimos 12 meses do ALFA", center=True)
                + c.grid(stats, cols=4)
                + mentor_markup
            ),
            variant="dark",
            waves="br",
        )
    )


def _render_estrutura() -> None:
    cards = [
        c.card(title=d["nome"], body=d["texto"], tag=d["frente"], step=i + 1)
        for i, d in enumerate(DIRETORIAS)
    ]
    c.render(
        c.section(
            c.container(
                c.section_head(
                    kicker="Estrutura organizacional",
                    title="Quatro diretorias, duas frentes",
                    subtitle="O ALFA Asset gere o fundo simulado; o ALFA Núcleo cuida das pessoas e da "
                    "ponte com o mercado.",
                    center=True,
                )
                + c.grid(cards, cols=4)
            ),
            variant="light",
        )
    )


def _render_plataforma_teaser(on_platform) -> None:
    with st.container(key="alfaband_dark_plataforma"):
        c.render(
            '<div class="alfa-split">'
            + c.reveal(
                c.eyebrow("Plataforma quant")
                + "<h2>As ferramentas de gestão, abertas</h2>"
                + c.lead(
                    "Fronteira eficiente de Markowitz, VaR/CVaR, CAPM, drawdown, Monte Carlo, "
                    "correlações e análise de ativos — os mesmos modelos que a diretoria de "
                    "Risco &amp; Quant usa para gerir o fundo, rodando ao vivo."
                ),
                step=1,
            )
            + c.reveal(
                c.grid(
                    [
                        c.kpi("Ferramentas", "6", note="Markowitz, CAPM, VaR/CVaR…"),
                        c.kpi("Ativos cobertos", "B3 + EUA", note="via Yahoo Finance"),
                        c.kpi("Benchmarks", "IBOV · CDI", note="atualizados sob demanda"),
                        c.kpi("Custo", "Aberto", note="uso livre e educacional"),
                    ],
                    cols=2,
                ),
                step=2,
            )
            + "</div>"
        )
        with c.cta_row("plataforma"):
            if st.button("Explorar a plataforma", key="cta_plataforma", type="primary"):
                on_platform()


def _render_cta_final(on_process) -> None:
    with st.container(key="alfaband_light_cta"):
        c.render(
            '<div class="alfa-center">'
            + c.section_head(
                title="Quer fazer parte?",
                subtitle="O processo seletivo acontece a cada semestre. Conheça a jornada do membro "
                "e os próximos passos.",
                center=True,
            )
            + "</div>"
        )
        with c.cta_row("psfinal"):
            if st.button("Ver processo seletivo", key="cta_ps_final", type="primary"):
                on_process()


def render(*, goto) -> None:
    portfolio_df, _ = fund_snapshot()
    _render_hero(lambda: goto("fundo"), lambda: goto("processo"))
    _render_ticker(portfolio_df)
    _render_quem_somos()
    _render_highlights()
    _render_estrutura()
    _render_plataforma_teaser(lambda: goto("plataforma"))
    _render_cta_final(lambda: goto("processo"))
