"""Página Newsletter — edições do Boletim ALFA, alimentada por content/newsletter.json.

Duas telas no mesmo arquivo:

* sem `?ed=` na URL, a lista: a edição mais recente em destaque e as anteriores
  em grade;
* com `?ed=<slug>`, a edição inteira lida dentro do site, desenhada com os
  componentes do tema.

Como publicar uma edição nova
-----------------------------
Acrescente uma entrada **no topo** da lista `edicoes` em
`content/newsletter.json`. Campos de vitrine: `numero`, `slug` (o que vai na
URL), `data`, `titulo`, `resumo`, `periodo`, `destaques`. Campos de leitura:
`carta`, `noticias` (tema, titulo, texto, fonte), `outros` (titulo, texto,
fonte) e `termos` (termo, definicao). `pdf` é opcional e, quando presente,
vira o botão de baixar.

O texto fica no JSON, não em PDF nem em HTML colado: assim a edição herda as
cores, as fontes e o comportamento no celular do resto do site.
"""

from __future__ import annotations

import streamlit as st

from site_pages._shared import load_content
from theme import components as c

PARAM_EDICAO = "ed"


# ------------------------------------------------------------------- vitrine


def _render_header() -> None:
    c.render(
        c.section(
            c.container(
                c.reveal(c.eyebrow("Publicações"), step=1)
                + c.reveal("<h1>Newsletter</h1>", step=2)
                + c.reveal(
                    c.lead(
                        "As edições da newsletter do ALFA: comentário de mercado, teses em "
                        "acompanhamento e o que a gestão do fundo aprendeu no período."
                    ),
                    step=3,
                )
            ),
            variant="dark",
            waves="tl",
            extra="alfa-section--tight",
        )
    )


def _link_edicao(edicao: dict) -> str:
    """URL interna da edição. O `target=_self` evita que o link abra outra aba."""
    return f'?p=newsletter&{PARAM_EDICAO}={edicao.get("slug", "")}'


def _botao_interno(rotulo: str, href: str, *, variant: str = "primary") -> str:
    return (
        f'<a class="alfa-btn alfa-btn--{variant}" href="{c.esc(href)}" '
        f'target="_self">{c.esc(rotulo)}</a>'
    )


def _render_destaque(edicao: dict) -> None:
    corpo = c.section_head(
        kicker=f'{edicao.get("numero", "")} · {edicao.get("data", "")}'.strip(" ·"),
        title=edicao.get("titulo", ""),
        subtitle=edicao.get("resumo", ""),
    )
    if edicao.get("periodo"):
        corpo += c.reveal(f'<p class="alfa-lead">{c.esc(edicao["periodo"])}</p>', step=2)
    if edicao.get("destaques"):
        corpo += c.reveal(c.tags(edicao["destaques"]), step=3)

    botoes = [_botao_interno("Ler no site", _link_edicao(edicao))]
    if edicao.get("pdf"):
        botoes.append(c.button("Baixar PDF", edicao["pdf"], variant="outline", new_tab=True))
    corpo += c.reveal(
        f'<div class="alfa-center" style="margin-top:28px">{c.ctas(*botoes)}</div>', step=4
    )
    c.render(c.section(c.container(corpo), variant="light"))


def _render_arquivo(edicoes: list[dict]) -> None:
    """As edições anteriores, em grade. Só aparece a partir da segunda."""
    if not edicoes:
        return
    cards = [
        c.card(
            title=f'{e.get("numero", "")} · {e.get("titulo", "")}'.strip(" ·"),
            body=f'{c.esc(e.get("resumo", ""))}<br>'
            f'<a href="{c.esc(_link_edicao(e))}" target="_self">Ler no site →</a>',
            tag=e.get("data", ""),
            step=(index % 6) + 1,
        )
        for index, e in enumerate(edicoes)
    ]
    c.render(
        c.section(
            c.container(
                c.section_head(kicker="Arquivo", title="Edições anteriores")
                + c.grid(cards, cols=3)
            ),
            variant="surface",
        )
    )


def _render_em_breve() -> None:
    c.render(
        c.section(
            c.container(
                c.coming_soon(
                    titulo="Primeira edição a caminho",
                    descricao="Estamos preparando esta página. Enquanto isso, acompanhe as "
                    "novidades do núcleo pelo Instagram.",
                )
                + '<div class="alfa-center" style="margin-top:28px">'
                + c.ctas(
                    c.button(
                        "Seguir @alfapucrio",
                        "https://www.instagram.com/alfapucrio",
                        variant="outline",
                        new_tab=True,
                    )
                )
                + "</div>"
            ),
            variant="light",
        )
    )


# --------------------------------------------------------------- edição lida


def _fonte(texto: str) -> str:
    return f'<p class="alfa-muted" style="font-size:.86rem">Fonte: {c.esc(texto)}</p>'


def _noticia(indice: int, item: dict) -> str:
    """Uma notícia numerada: tema como kicker, manchete, corpo e fonte."""
    return (
        '<div style="margin-bottom:clamp(36px,5vw,56px)">'
        + c.section_head(
            kicker=f'{indice:02d} · {item.get("tema", "")}'.strip(" ·"),
            title=item.get("titulo", ""),
            level=3,
        )
        + f'<p>{c.esc(item.get("texto", ""))}</p>'
        + (_fonte(item["fonte"]) if item.get("fonte") else "")
        + "</div>"
    )


def _cabecalho_edicao(edicao: dict) -> None:
    c.render(
        c.section(
            c.container(
                c.reveal(
                    c.eyebrow(f'{edicao.get("numero", "")} · {edicao.get("data", "")}'.strip(" ·")),
                    step=1,
                )
                + c.reveal(f'<h1>{c.esc(edicao.get("titulo", ""))}</h1>', step=2)
                + (c.reveal(c.lead(edicao["periodo"]), step=3) if edicao.get("periodo") else "")
            ),
            variant="dark",
            waves="tl",
            extra="alfa-section--tight",
        )
    )


def _rodape_edicao(edicao: dict) -> None:
    botoes = [_botao_interno("Ver todas as edições", "?p=newsletter", variant="outline")]
    if edicao.get("pdf"):
        botoes.insert(0, c.button("Baixar PDF", edicao["pdf"], new_tab=True))
    c.render(
        c.section(
            c.container(f'<div class="alfa-center">{c.ctas(*botoes)}</div>'),
            variant="light",
        )
    )


def _render_edicao(edicao: dict) -> None:
    _cabecalho_edicao(edicao)

    if edicao.get("carta"):
        c.render(
            c.section(
                c.container(
                    c.section_head(kicker="Carta ao leitor", title="O que pautou a semana")
                    + f'<p class="alfa-lead">{c.esc(edicao["carta"])}</p>'
                ),
                variant="light",
            )
        )

    noticias = edicao.get("noticias", [])
    if noticias:
        c.render(
            c.section(
                c.container(
                    c.section_head(
                        kicker=f"{len(noticias)} notícias", title="As notícias da semana"
                    )
                    + "".join(_noticia(i, item) for i, item in enumerate(noticias, start=1))
                ),
                variant="surface",
            )
        )

    outros = edicao.get("outros", [])
    if outros:
        cards = [
            c.card(
                title=item.get("titulo", ""),
                body=c.esc(item.get("texto", ""))
                + (_fonte(item["fonte"]) if item.get("fonte") else ""),
                step=(i % 6) + 1,
            )
            for i, item in enumerate(outros)
        ]
        c.render(
            c.section(
                c.container(
                    c.section_head(kicker="Também na semana", title="Outros destaques")
                    + c.grid(cards, cols=2)
                ),
                variant="light",
            )
        )

    termos = edicao.get("termos", [])
    if termos:
        cards = [
            c.card(title=item.get("termo", ""), body=c.esc(item.get("definicao", "")), step=(i % 6) + 1)
            for i, item in enumerate(termos)
        ]
        c.render(
            c.section(
                c.container(
                    c.section_head(
                        kicker="Glossário",
                        title="Termos para estudar",
                        subtitle="O vocabulário que apareceu nas notícias desta edição.",
                    )
                    + c.grid(cards, cols=2)
                ),
                variant="surface",
            )
        )

    _rodape_edicao(edicao)


# ------------------------------------------------------------------- roteiro


def render(*, goto=None) -> None:  # noqa: ARG001 — assinatura comum às páginas
    edicoes = load_content("newsletter").get("edicoes", [])

    pedida = st.query_params.get(PARAM_EDICAO)
    atual = next((e for e in edicoes if str(e.get("slug", "")) == str(pedida)), None)
    if atual is not None:
        _render_edicao(atual)
        return

    _render_header()
    if not edicoes:
        _render_em_breve()
        return
    _render_destaque(edicoes[0])
    _render_arquivo(edicoes[1:])
