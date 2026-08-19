"""Página Alumni — placeholder, a preencher.

Como preencher
--------------
1. Crie `content/alumni.json` no formato:

       {
         "alumni": [
           {
             "nome": "Nome Sobrenome",
             "turma": "2024.1",
             "cargo": "Analista · Nome da Gestora",
             "foto": "",
             "linkedin": ""
           }
         ]
       }

   As fotos vão em `assets/membros/` (mesma pasta dos membros ativos).

2. Descomente o bloco `_render_alumni()` no fim deste arquivo e a chamada dele
   em `render()`. O layout de grade já é o mesmo da página Membros.
"""

from __future__ import annotations

from theme import components as c

# from site_pages._shared import load_content, photo_uri  # usar ao ligar o JSON


def _render_header() -> None:
    c.render(
        c.section(
            c.container(
                c.reveal(c.eyebrow("Rede ALFA"), step=1)
                + c.reveal("<h1>Alumni</h1>", step=2)
                + c.reveal(
                    c.lead(
                        "Quem passou pelo ALFA e hoje atua no mercado — a rede que conecta as "
                        "turmas do núcleo ao buy-side, ao sell-side e à academia."
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
                    titulo="Estamos montando a rede",
                    descricao="Esta página vai reunir os ex-membros do ALFA por turma. "
                    "Se você passou pelo núcleo, fale com a gente.",
                )
                + '<div class="alfa-center" style="margin-top:28px">'
                + c.ctas(
                    c.button(
                        "Falar com o ALFA", "mailto:alfapucrio@gmail.com", variant="outline"
                    )
                )
                + "</div>"
            ),
            variant="light",
        )
    )


# ---------------------------------------------------------------------------
# Bloco pronto para quando a lista existir — descomente e chame
# `_render_alumni()` dentro de `render()`, logo após `_render_header()`.
# ---------------------------------------------------------------------------
# def _render_alumni() -> None:
#     pessoas = load_content("alumni").get("alumni", [])
#     if not pessoas:
#         return
#     turmas: dict[str, list[dict]] = {}
#     for pessoa in pessoas:
#         turmas.setdefault(pessoa.get("turma", "Sem turma"), []).append(pessoa)
#
#     blocos = []
#     for turma in sorted(turmas, reverse=True):
#         cards = [
#             c.member(
#                 nome=p.get("nome", ""),
#                 cargo=p.get("cargo", ""),
#                 foto_uri=photo_uri("membros", p.get("foto", "")),
#                 linkedin=p.get("linkedin", ""),
#                 step=(index % 6) + 1,
#             )
#             for index, p in enumerate(turmas[turma])
#         ]
#         blocos.append(
#             '<div style="margin-bottom:clamp(44px,6vw,72px)">'
#             + c.section_head(kicker=f"Turma {turma}", title=f"{len(cards)} alumni")
#             + c.grid(cards, cols=4)
#             + "</div>"
#         )
#     c.render(c.section(c.container("".join(blocos)), variant="light"))
