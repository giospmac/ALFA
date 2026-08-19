"""Blocos visuais em HTML do site ALFA.

Cada função devolve uma string de HTML já compactada (sem quebras de linha com
indentação, que o parser de markdown do Streamlit interpretaria como bloco de
código). Use `render()` para escrever direto na página.
"""

from __future__ import annotations

import html as _html
import re
from contextlib import contextmanager
from typing import Iterable, Iterator, Sequence

import streamlit as st

from theme.styles import asset_data_uri

# ---------------------------------------------------------------- utilidades


def esc(value: object) -> str:
    """Escapa texto vindo dos JSONs de conteúdo."""
    return _html.escape(str(value), quote=True)


def compact(markup: str) -> str:
    """Colapsa quebras de linha e indentação, que o markdown do Streamlit
    interpretaria como bloco de código.

    Vira um espaço (e não string vazia) para não colar palavras que estavam
    em linhas diferentes. Espaço entre tags é ignorado em flex/grid.
    """
    return re.sub(r"\n\s*", " ", markup).strip()


def render(markup: str) -> None:
    st.markdown(compact(markup), unsafe_allow_html=True)


def _classes(*names: str | None) -> str:
    return " ".join(n for n in names if n)


@contextmanager
def cta_row(name: str) -> Iterator[None]:
    """Linha de botões reais do Streamlit, centralizada e lado a lado.

    Usa um container com `key` em vez de `st.columns` porque colunas criam
    alturas sobrando dentro das faixas institucionais.
    """
    with st.container(key=f"alfa_cta_{name}"):
        yield


# ------------------------------------------------------------------ estrutura


def section(inner: str, *, variant: str = "light", waves: str = "", extra: str = "", anchor: str = "") -> str:
    """Faixa full-bleed. `variant`: light | surface | dark."""
    wave_markup = ""
    if waves:
        wave_markup = f'<div class="alfa-waves {"alfa-waves--br" if waves == "br" else ""}"></div>'
    anchor_markup = f'<span id="{esc(anchor)}"></span>' if anchor else ""
    return (
        f'<section class="{_classes("alfa-section", f"alfa-section--{variant}", extra)}">'
        f"{anchor_markup}{wave_markup}{inner}</section>"
    )


def container(inner: str, *, narrow: bool = False, extra: str = "") -> str:
    return f'<div class="{_classes("alfa-container", "alfa-container--narrow" if narrow else None, extra)}">{inner}</div>'


def grid(items: Iterable[str], *, cols: int = 3, extra: str = "") -> str:
    body = "".join(items)
    return f'<div class="{_classes("alfa-grid", f"alfa-grid--{cols}", extra)}">{body}</div>'


def reveal(inner: str, *, step: int = 1, tag: str = "div", extra: str = "") -> str:
    step = max(1, min(step, 6))
    return f'<{tag} class="{_classes("alfa-reveal", f"alfa-reveal-{step}", extra)}">{inner}</{tag}>'


# ---------------------------------------------------------------- tipografia


def eyebrow(text: str) -> str:
    return f'<span class="alfa-eyebrow">{esc(text)}</span>'


def lead(text: str) -> str:
    return f'<p class="alfa-lead">{text}</p>'


def section_head(
    *, kicker: str = "", title: str = "", subtitle: str = "", center: bool = False, level: int = 2
) -> str:
    parts = []
    if kicker:
        parts.append(eyebrow(kicker))
    if title:
        parts.append(f"<h{level}>{title}</h{level}>")
    if subtitle:
        parts.append(lead(subtitle))
    wrapper_class = "alfa-center" if center else ""
    return f'<div class="{_classes(wrapper_class, "alfa-reveal")}" style="margin-bottom:clamp(28px,4vw,44px)">{"".join(parts)}</div>'


# ---------------------------------------------------------------- componentes


def button(label: str, href: str, *, variant: str = "primary", new_tab: bool = False) -> str:
    target = ' target="_blank" rel="noopener"' if new_tab else ""
    return f'<a class="alfa-btn alfa-btn--{variant}" href="{esc(href)}"{target}>{esc(label)}</a>'


def ctas(*buttons: str, center: bool = True) -> str:
    style = "" if center else ' style="justify-content:flex-start"'
    return f'<div class="alfa-ctas"{style}>{"".join(buttons)}</div>'


def card(*, title: str, body: str, tag: str = "", step: int = 1) -> str:
    tag_markup = f'<span class="alfa-card__tag">{esc(tag)}</span>' if tag else ""
    return reveal(
        f'<div class="alfa-card">{tag_markup}<h3>{esc(title)}</h3><p>{body}</p></div>',
        step=step,
    )


def stat(value: str, label: str, *, step: int = 1) -> str:
    return reveal(
        f'<div class="alfa-stat"><div class="alfa-stat__value">{esc(value)}</div>'
        f'<div class="alfa-stat__label">{esc(label)}</div></div>',
        step=step,
    )


def tags(items: Sequence[str]) -> str:
    return "".join(f'<span class="alfa-tag">{esc(i)}</span>' for i in items)


def chip(text: str, *, tone: str = "info") -> str:
    return f'<span class="alfa-chip alfa-chip--{tone}">{esc(text)}</span>'


def timeline(steps: Sequence[dict]) -> str:
    rows = []
    for index, step in enumerate(steps, start=1):
        rows.append(
            f'<li class="alfa-tl-step alfa-reveal alfa-reveal-{min(index, 6)}" data-step="{index}">'
            f'<h3>{esc(step["titulo"])}</h3><p>{esc(step["descricao"])}</p></li>'
        )
    return f'<ol class="alfa-timeline">{"".join(rows)}</ol>'


def kpi(label: str, value: str, *, note: str = "", tone: str = "", step: int = 1) -> str:
    note_markup = f'<div class="alfa-kpi__note {tone}">{esc(note)}</div>' if note else ""
    return reveal(
        f'<div class="alfa-kpi"><div class="alfa-kpi__label">{esc(label)}</div>'
        f'<div class="alfa-kpi__value">{esc(value)}</div>{note_markup}</div>',
        step=step,
    )


def coming_soon(*, titulo: str, descricao: str, chip_texto: str = "Em breve") -> str:
    """Estado vazio de uma página ainda sem conteúdo publicado."""
    return reveal(
        '<div class="alfa-empty">'
        f'<div style="margin-bottom:14px">{chip(chip_texto)}</div>'
        f"<h3>{esc(titulo)}</h3><p>{esc(descricao)}</p></div>"
    )


def member(*, nome: str, cargo: str, foto_uri: str = "", linkedin: str = "", step: int = 1) -> str:
    if foto_uri:
        avatar = f'<img src="{foto_uri}" alt="{esc(nome)}">'
    else:
        initials = "".join(part[0] for part in nome.split()[:2]).upper() or "A"
        avatar = f'<div class="alfa-member__avatar">{esc(initials)}</div>'
    name_markup = esc(nome)
    if linkedin:
        name_markup = f'<a href="{esc(linkedin)}" target="_blank" rel="noopener">{name_markup}</a>'
    return reveal(
        f'<div class="alfa-member">{avatar}<div class="alfa-member__name">{name_markup}</div>'
        f'<div class="alfa-member__role">{esc(cargo)}</div></div>',
        step=step,
    )


def alumni_card(*, nome: str, posicao: str, linkedin: str = "", step: int = 1) -> str:
    """Card compacto de alumni: nome, posição e o espaço do LinkedIn.

    O slot do ícone é sempre renderizado — invisível quando não há link — para
    que todos os cards tenham a mesma largura útil e nada se desloque conforme
    os links forem preenchidos em `content/alumni.json`.
    """
    nome_markup = esc(nome)
    if linkedin:
        nome_markup = f'<a href="{esc(linkedin)}" target="_blank" rel="noopener">{nome_markup}</a>'
        slot = (
            f'<a class="alfa-alum__link" href="{esc(linkedin)}" target="_blank" rel="noopener" '
            f'title="LinkedIn de {esc(nome)}" aria-label="LinkedIn de {esc(nome)}">{_LINKEDIN}</a>'
        )
    else:
        slot = '<span class="alfa-alum__link alfa-alum__link--vazio" aria-hidden="true"></span>'
    return reveal(
        f'<div class="alfa-alum"><div class="alfa-alum__info">'
        f'<div class="alfa-alum__name">{nome_markup}</div>'
        f'<div class="alfa-alum__role">{esc(posicao)}</div></div>{slot}</div>',
        step=step,
    )


def agenda(rows: Sequence[dict]) -> str:
    items = []
    for row in rows:
        cls = "alfa-agenda__title alfa-agenda__title--hl" if row.get("destaque") else "alfa-agenda__title"
        items.append(
            f'<div class="alfa-agenda__row"><div class="alfa-agenda__date">{esc(row["data"])}</div>'
            f'<div class="{cls}">{esc(row["titulo"])}</div></div>'
        )
    return f'<div class="alfa-agenda">{"".join(items)}</div>'


def data_table(headers: Sequence[str], rows: Sequence[Sequence[str]], *, numeric_from: int = 1) -> str:
    head = "".join(
        f'<th class="{"num" if i >= numeric_from else ""}">{esc(h)}</th>' for i, h in enumerate(headers)
    )
    body = []
    for row in rows:
        cells = "".join(
            f'<td class="{"num" if i >= numeric_from else ""}">{cell}</td>' for i, cell in enumerate(row)
        )
        body.append(f"<tr>{cells}</tr>")
    return (
        f'<div class="alfa-table-wrap"><table class="alfa-table"><thead><tr>{head}</tr></thead>'
        f'<tbody>{"".join(body)}</tbody></table></div>'
    )


# ------------------------------------------------------------------ hero/nav


def brand(*, logo: str = "logo-alfa-white.png") -> str:
    return (
        f'<div class="alfa-brand"><img src="{asset_data_uri(logo)}" alt="ALFA">'
        f'<div><div class="alfa-brand-txt">ALFA</div>'
        f'<div class="alfa-brand-sub">PUC-Rio</div></div></div>'
    )


# -------------------------------------------------------------------- rodapé

_INSTAGRAM = (
    '<svg viewBox="0 0 24 24"><path d="M12 2.16c3.2 0 3.58.01 4.85.07 3.25.15 4.77 1.69 4.92 4.92.06 1.27.07 1.64.07 4.85s-.01 3.58-.07 4.85c-.15 3.23-1.66 4.77-4.92 4.92-1.27.06-1.64.07-4.85.07s-3.58-.01-4.85-.07c-3.26-.15-4.77-1.7-4.92-4.92C2.17 15.58 2.16 15.2 2.16 12s.01-3.58.07-4.85c.15-3.23 1.66-4.77 4.92-4.92C8.42 2.17 8.8 2.16 12 2.16zm0-2.16C8.74 0 8.33.01 7.05.07 2.7.27.28 2.69.08 7.05.01 8.33 0 8.74 0 12s.01 3.67.07 4.95c.2 4.36 2.62 6.78 6.98 6.98C8.33 23.99 8.74 24 12 24s3.67-.01 4.95-.07c4.35-.2 6.78-2.62 6.98-6.98.06-1.28.07-1.69.07-4.95s-.01-3.67-.07-4.95c-.2-4.35-2.62-6.78-6.98-6.98C15.67.01 15.26 0 12 0zm0 5.84a6.16 6.16 0 100 12.32 6.16 6.16 0 000-12.32zm0 10.16a4 4 0 110-8 4 4 0 010 8zm6.41-11.85a1.44 1.44 0 100 2.88 1.44 1.44 0 000-2.88z"/></svg>'
)
# Glifo com o quadrado e o "in" vazado: a cor vem da paleta ALFA (o vazado
# mostra o fundo do card), nunca do azul da marca.
_LINKEDIN = (
    '<svg viewBox="0 0 24 24"><path d="M19 0H5a5 5 0 00-5 5v14a5 5 0 005 5h14a5 5 0 005-5V5a5 5 0 00-5-5zM8 19H5V8h3v11zM6.5 6.73a1.77 1.77 0 110-3.53 1.77 1.77 0 010 3.53zM20 19h-3v-5.6c0-3.37-4-3.11-4 0V19h-3V8h3v1.77c1.4-2.59 7-2.78 7 2.47V19z"/></svg>'
)
_MAIL = (
    '<svg viewBox="0 0 24 24"><path d="M20 4H4a2 2 0 00-2 2v12a2 2 0 002 2h16a2 2 0 002-2V6a2 2 0 00-2-2zm0 4.24l-8 4.76-8-4.76V6l8 4.76L20 6v2.24z"/></svg>'
)


def footer_html(links_markup: str = "") -> str:
    """Rodapé do site. `links_markup` é o <li> de navegação já montado."""
    links = links_markup
    return compact(
        f"""
        <footer class="alfa-footer"><div class="alfa-container">
          <div class="alfa-footer__grid">
            <div>
              {brand()}
              <p style="margin-top:14px;max-width:34rem">Laboratório de Finanças Aplicadas do Departamento de
              Economia da PUC-Rio. Fundo simulado, equity research e finanças quantitativas.</p>
              <div class="alfa-social">
                <a href="https://www.instagram.com/alfapucrio" target="_blank" rel="noopener" title="Instagram">{_INSTAGRAM}</a>
                <a href="https://www.linkedin.com/company/alfapucrio" target="_blank" rel="noopener" title="LinkedIn">{_LINKEDIN}</a>
                <a href="mailto:alfapucrio@gmail.com" title="E-mail">{_MAIL}</a>
              </div>
            </div>
            <div><h4>Navegação</h4><ul>{links}</ul></div>
            <div><h4>Contato</h4><ul>
              <li><a href="mailto:alfapucrio@gmail.com">alfapucrio@gmail.com</a></li>
              <li><a href="https://www.instagram.com/alfapucrio" target="_blank" rel="noopener">@alfapucrio</a></li>
              <li>PUC-Rio · Gávea, Rio de Janeiro</li>
            </ul></div>
          </div>
          <div class="alfa-footer__bottom">
            <span>© 2026 ALFA — Laboratório de Finanças Aplicadas PUC-Rio</span>
            <span>Conteúdo educacional. Nada aqui constitui recomendação de investimento.</span>
          </div>
        </div></footer>
        """
    )
