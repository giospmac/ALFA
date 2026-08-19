"""Página Newsletter — placeholder, a preencher.

Como preencher
--------------
1. Crie `content/newsletter.json` no formato:

       {
         "edicoes": [
           {
             "numero": "#01",
             "data": "Março de 2026",
             "titulo": "Título da edição",
             "resumo": "Uma ou duas linhas sobre o conteúdo.",
             "link": "https://…"
           }
         ]
       }

2. Descomente o bloco `_render_edicoes()` no fim deste arquivo e a chamada
   dele em `render()`. O resto (cabeçalho, layout, cores) já está pronto.
"""

from __future__ import annotations

from theme import components as c

# from site_pages._shared import load_content  # usar ao ligar o JSON


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


def render(*, goto=None) -> None:  # noqa: ARG001 — assinatura comum às páginas
    _render_header()
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


# ---------------------------------------------------------------------------
# Bloco pronto para quando houver edições publicadas — basta descomentar e
# chamar `_render_edicoes()` dentro de `render()`, logo após `_render_header()`.
# ---------------------------------------------------------------------------
# def _render_edicoes() -> None:
#     edicoes = load_content("newsletter").get("edicoes", [])
#     if not edicoes:
#         return
#     cards = [
#         c.card(
#             title=f'{e["numero"]} · {e["titulo"]}',
#             body=f'{c.esc(e["resumo"])}<br><a href="{c.esc(e["link"])}" target="_blank" '
#             f'rel="noopener">Ler edição →</a>',
#             tag=e.get("data", ""),
#             step=(index % 6) + 1,
#         )
#         for index, e in enumerate(edicoes)
#     ]
#     c.render(
#         c.section(
#             c.container(
#                 c.section_head(kicker="Arquivo", title="Todas as edições")
#                 + c.grid(cards, cols=3)
#             ),
#             variant="light",
#         )
#     )
