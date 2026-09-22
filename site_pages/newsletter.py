"""Página Newsletter — edições do Boletim ALFA, alimentada por content/newsletter.json.

Como publicar uma edição nova
-----------------------------
1. Coloque o PDF em `static/newsletter/` (o Streamlit serve essa pasta porque
   `.streamlit/config.toml` tem `enableStaticServing`).
2. Acrescente uma entrada **no topo** da lista `edicoes` em
   `content/newsletter.json`, com `link` apontando para
   `app/static/newsletter/<arquivo>.pdf`.

A primeira entrada da lista vira o destaque do topo; as demais caem no arquivo,
em grade. Sem nenhuma edição, a página volta ao aviso de "em breve".
"""

from __future__ import annotations

from site_pages._shared import load_content
from theme import components as c


def _render_header() -> None:
    c.render(
        c.section(
            c.container(
                c.reveal(c.eyebrow("Publicações"), step=1)
                + c.reveal("<h1>Newsletter</h1>", step=2)
                + c.reveal(
                    c.lead(
                        "As edições da newsletter do ALFA — comentário de mercado, teses em "
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


def _render_destaque(edicao: dict) -> None:
    """A edição mais recente, em bloco largo com o botão de leitura."""
    periodo = edicao.get("periodo", "")
    destaques = edicao.get("destaques", [])
    corpo = c.section_head(
        kicker=f'{edicao.get("numero", "")} · {edicao.get("data", "")}'.strip(" ·"),
        title=edicao.get("titulo", ""),
        subtitle=edicao.get("resumo", ""),
    )
    if periodo:
        corpo += c.reveal(f'<p class="alfa-lead">{c.esc(periodo)}</p>', step=2)
    if destaques:
        corpo += c.reveal(c.tags(destaques), step=3)
    corpo += c.reveal(
        '<div class="alfa-center" style="margin-top:28px">'
        + c.ctas(c.button("Ler edição (PDF)", edicao.get("link", ""), new_tab=True))
        + "</div>",
        step=4,
    )
    c.render(c.section(c.container(corpo), variant="light"))


def _render_arquivo(edicoes: list[dict]) -> None:
    """As edições anteriores, em grade. Só aparece a partir da segunda."""
    if not edicoes:
        return
    cards = [
        c.card(
            title=f'{e.get("numero", "")} · {e.get("titulo", "")}'.strip(" ·"),
            body=f'{c.esc(e.get("resumo", ""))}<br><a href="{c.esc(e.get("link", ""))}" '
            'target="_blank" rel="noopener">Ler edição →</a>',
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


def render(*, goto=None) -> None:  # noqa: ARG001 — assinatura comum às páginas
    _render_header()

    edicoes = load_content("newsletter").get("edicoes", [])
    if not edicoes:
        _render_em_breve()
        return

    _render_destaque(edicoes[0])
    _render_arquivo(edicoes[1:])
