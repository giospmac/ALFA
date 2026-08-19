"""ALFA — Laboratório de Finanças Aplicadas PUC-Rio.

Site institucional + plataforma quantitativa em um único app Streamlit.

Arquitetura
-----------
    app.py            casca: config, CSS, topbar, roteador, rodapé
    theme/            design system (tokens, CSS, componentes HTML, Plotly)
    site_pages/       páginas institucionais (conteúdo em `content/*.json`)
    ui/               telas da plataforma (modelos e dashboards)
    services/ core/   cálculos financeiros e acesso a dados
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
_ICON = ROOT / "assets" / "logo-alfa-blue.png"

st.set_page_config(
    page_title="ALFA — Laboratório de Finanças Aplicadas PUC-Rio",
    page_icon=str(_ICON) if _ICON.is_file() else "📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

from site_pages import (  # noqa: E402
    alumni,
    atividades,
    fundo,
    inicio,
    membros,
    newsletter,
    plataforma,
    processo,
)
from theme import components as c  # noqa: E402
from theme.plotly_theme import register_templates  # noqa: E402
from theme.styles import inject_global_css  # noqa: E402

NAV_KEY = "alfa_nav_pills"
STATE_KEY = "page"
PENDING_KEY = "_pending_page"

#: Ordem das seções no menu — mover uma linha aqui reordena a navegação
#: (e o rodapé) sem mexer em mais nada.
PAGES: dict[str, tuple[str, object]] = {
    "inicio": ("Início", inicio),
    "fundo": ("O Fundo", fundo),
    "plataforma": ("Plataforma", plataforma),
    "newsletter": ("Newsletter", newsletter),
    "membros": ("Membros", membros),
    "alumni": ("Alumni", alumni),
    "atividades": ("Atividades", atividades),
    "processo": ("Processo Seletivo", processo),
}
LABELS = {key: label for key, (label, _) in PAGES.items()}
DEFAULT_PAGE = "inicio"


# --------------------------------------------------------------------- rotas


def _current_page() -> str:
    page = st.session_state.get(STATE_KEY)
    return page if page in PAGES else DEFAULT_PAGE


def _set_page(page: str) -> None:
    if page not in PAGES:
        page = DEFAULT_PAGE
    st.session_state[STATE_KEY] = page
    st.session_state[NAV_KEY] = LABELS[page]
    st.query_params["p"] = page


def goto(page: str) -> None:
    """Navega para outra seção (usado pelos CTAs das páginas).

    A troca é agendada e aplicada no início do próximo rerun: o Streamlit não
    permite alterar o `session_state` de um widget (as pílulas do menu) depois
    que ele já foi instanciado nesta execução.
    """
    if page == _current_page() or page not in PAGES:
        return
    st.session_state[PENDING_KEY] = page
    st.rerun()


def _bootstrap_route() -> None:
    """Aplica navegação pendente e deixa a URL mandar no resto.

    Na ordem: (1) a seção pedida por um CTA na execução anterior; (2) o `?p=` da
    URL sempre que divergir do estado — é o que faz deep link e o botão voltar do
    navegador funcionarem, já que no reload o Streamlit reaproveita a sessão e o
    estado sobreviveria à troca de URL; (3) o padrão, na primeira execução.

    Não há conflito com os cliques na navegação: toda troca de seção grava o
    parâmetro junto com o estado, então os dois só divergem quando a URL muda
    por fora.
    """
    pending = st.session_state.pop(PENDING_KEY, None)
    if pending:
        _set_page(pending)
        return
    requested = st.query_params.get("p")
    if requested in PAGES and requested != st.session_state.get(STATE_KEY):
        _set_page(requested)
    elif STATE_KEY not in st.session_state:
        _set_page(DEFAULT_PAGE)


def _on_nav_change() -> None:
    label = st.session_state.get(NAV_KEY)
    if label is None:  # clicar na pílula ativa desmarca; mantemos a seção atual
        st.session_state[NAV_KEY] = LABELS[_current_page()]
        return
    page = next((key for key, value in LABELS.items() if value == label), DEFAULT_PAGE)
    st.session_state[STATE_KEY] = page
    st.query_params["p"] = page


# ------------------------------------------------------------------- chrome


def _render_topbar() -> None:
    with st.container(key="alfa_topnav"):
        st.markdown(c.compact(c.brand()), unsafe_allow_html=True)
        st.pills(
            "Seções",
            list(LABELS.values()),
            key=NAV_KEY,
            label_visibility="collapsed",
            on_change=_on_nav_change,
        )


def _render_footer() -> None:
    links = "".join(
        f'<li><a href="?p={key}" target="_self">{c.esc(label)}</a></li>'
        for key, label in LABELS.items()
    )
    st.markdown(c.footer_html(links), unsafe_allow_html=True)


# --------------------------------------------------------------------- main


def main() -> None:
    inject_global_css()
    register_templates()
    _bootstrap_route()

    _render_topbar()

    page = _current_page()
    with st.container(key="alfa_root"):
        PAGES[page][1].render(goto=goto)
        _render_footer()


if __name__ == "__main__":
    main()
