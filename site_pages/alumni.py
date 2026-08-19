"""Página Alumni — ex-membros do núcleo, agrupados por posição.

Como editar
-----------
A lista vive em `content/alumni.json`. Cada entrada tem `nome`, `posicao` e
`linkedin`. Todo card reserva o espaço do ícone do LinkedIn à direita: com a URL
preenchida o ícone aparece e o nome também vira clicável; vazio, o espaço fica
invisível — assim a grade não muda de forma conforme os links chegam.

O agrupamento é derivado da `posicao` pelas palavras-chave em `GRUPOS`: quem
tiver "Presidente" ou "VP" cai em Presidência, "Diretor"/"Diretora" em
Diretoria, e o restante em Associados. Uma posição nova que não case com
nenhuma palavra-chave vai para o último grupo.
"""

from __future__ import annotations

import unicodedata

import streamlit as st

from site_pages._shared import load_content
from theme import components as c

#: (título do grupo, palavras-chave da posição). A ordem é a de exibição e a
#: última entrada funciona como grupo de fallback.
GRUPOS: list[tuple[str, tuple[str, ...], str]] = [
    (
        "Presidência",
        ("presidente", "vp", "vice-presidente"),
        "Quem liderou o núcleo e as duas frentes, ALFA Asset e ALFA Núcleo.",
    ),
    (
        "Diretoria",
        ("diretor", "diretora"),
        "Ex-diretores das áreas de Equity Research, Gestão & Risco, Mercado e Pessoas.",
    ),
    (
        "Associados",
        (),
        "Quem passou pelas análises, pelos modelos e pelos projetos do núcleo.",
    ),
]


def _normalizar(texto: str) -> str:
    sem_acento = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode()
    return sem_acento.lower()


def _agrupar(pessoas: list[dict]) -> dict[str, list[dict]]:
    grupos: dict[str, list[dict]] = {titulo: [] for titulo, _, _ in GRUPOS}
    fallback = GRUPOS[-1][0]
    for pessoa in pessoas:
        posicao = _normalizar(pessoa.get("posicao", ""))
        destino = next(
            (titulo for titulo, chaves, _ in GRUPOS if any(chave in posicao for chave in chaves)),
            fallback,
        )
        grupos[destino].append(pessoa)
    return grupos


def _render_header(total: int) -> None:
    contagem = f"{total} ex-membros" if total else "Os ex-membros do núcleo"
    c.render(
        c.section(
            c.container(
                c.reveal(c.eyebrow("Rede ALFA"), step=1)
                + c.reveal("<h1>Alumni</h1>", step=2)
                + c.reveal(
                    c.lead(
                        f"{contagem} que construíram o ALFA desde a fundação — de presidências e "
                        "diretorias a associados que passaram pelas análises e pelos modelos do fundo."
                    ),
                    step=3,
                )
            ),
            variant="dark",
            waves="tl",
            extra="alfa-section--tight",
        )
    )


def _render_grupo(titulo: str, descricao: str, pessoas: list[dict], *, variant: str) -> None:
    if not pessoas:
        return
    cards = [
        c.alumni_card(
            nome=pessoa.get("nome", ""),
            posicao=pessoa.get("posicao", ""),
            linkedin=pessoa.get("linkedin", ""),
            step=(index % 6) + 1,
        )
        for index, pessoa in enumerate(pessoas)
    ]
    c.render(
        c.section(
            c.container(
                c.section_head(
                    kicker=f"{len(pessoas)} {'pessoa' if len(pessoas) == 1 else 'pessoas'}",
                    title=titulo,
                    subtitle=descricao,
                )
                + c.grid(cards, cols=3)
            ),
            variant=variant,
        )
    )


def _render_cta() -> None:
    with st.container(key="alfaband_dark_alumnicta"):
        c.render(
            '<div class="alfa-center">'
            + c.section_head(
                title="Passou pelo ALFA e não está aqui?",
                subtitle="Nos escreva para entrar na lista — ou para mandar seu LinkedIn e "
                "aparecer com o link no seu nome.",
                center=True,
            )
            + c.ctas(c.button("Falar com o ALFA", "mailto:alfapucrio@gmail.com"))
            + "</div>"
        )


def render(*, goto=None) -> None:  # noqa: ARG001 — assinatura comum às páginas
    pessoas = load_content("alumni").get("alumni", [])
    _render_header(len(pessoas))

    if not pessoas:
        c.render(
            c.section(
                c.container(
                    c.coming_soon(
                        titulo="Estamos montando a rede",
                        descricao="Edite content/alumni.json para publicar a lista de ex-membros.",
                    )
                ),
                variant="light",
            )
        )
        _render_cta()
        return

    grupos = _agrupar(pessoas)
    for indice, (titulo, _, descricao) in enumerate(GRUPOS):
        _render_grupo(
            titulo,
            descricao,
            grupos[titulo],
            variant="light" if indice % 2 == 0 else "surface",
        )

    _render_cta()
