"""Templates Plotly da identidade ALFA.

Registra dois templates globais — `alfa` (fundo claro) e `alfa_dark` (faixas
institucionais escuras) — e os define como padrão. Todos os gráficos do app,
inclusive os herdados das páginas antigas, passam a nascer estilizados sem
precisar de `_apply_alfa_style` em cada arquivo.
"""

from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio

from theme import tokens as T

_BASE_FONT = dict(family=T.FONT_STACK, size=12)


def _template(*, dark: bool) -> go.layout.Template:
    ink = T.ON_DARK if dark else T.INK
    soft = T.ON_DARK_SOFT if dark else T.INK_SOFT
    grid = "rgba(138,168,250,.14)" if dark else "rgba(20,25,46,.07)"
    paper = "rgba(0,0,0,0)" if dark else T.SURFACE
    hover_bg = T.NAVY_900 if dark else T.SURFACE

    axis = dict(
        showgrid=True,
        gridcolor=grid,
        gridwidth=1,
        zeroline=False,
        showline=False,
        ticks="outside",
        ticklen=4,
        tickcolor="rgba(0,0,0,0)",
        tickfont=dict(color=soft, size=11),
        title=dict(font=dict(color=soft, size=11.5)),
        automargin=True,
    )

    return go.layout.Template(
        layout=dict(
            font={**_BASE_FONT, "color": soft},
            paper_bgcolor=paper,
            plot_bgcolor=paper,
            colorway=T.CATEGORICAL,
            colorscale=dict(
                sequential=[[i / (len(T.SEQUENTIAL) - 1), c] for i, c in enumerate(T.SEQUENTIAL)],
                diverging=[[i / (len(T.DIVERGING) - 1), c] for i, c in enumerate(T.DIVERGING)],
            ),
            title=dict(
                font=dict(color=ink, size=15, family=T.FONT_STACK, weight=700),
                x=0.01,
                xanchor="left",
                y=0.96,
            ),
            margin=dict(l=8, r=14, t=52, b=56),
            hoverlabel=dict(
                bgcolor=hover_bg,
                bordercolor="rgba(138,168,250,.35)" if dark else T.BORDER,
                font=dict(color=T.ON_DARK if dark else T.INK, family=T.FONT_STACK, size=12),
            ),
            hovermode="x unified",
            # legenda embaixo: no mobile o topo já é do título e do eixo
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.16,
                xanchor="center",
                x=0.5,
                bgcolor="rgba(0,0,0,0)",
                font=dict(color=soft, size=11),
                title=dict(text=""),
            ),
            xaxis=axis,
            yaxis=axis,
            dragmode="pan",
            separators=",.",
        )
    )


def register_templates() -> None:
    """Registra e ativa o template ALFA como padrão do Plotly."""
    pio.templates["alfa"] = _template(dark=False)
    pio.templates["alfa_dark"] = _template(dark=True)
    pio.templates.default = "alfa"


def style(fig: go.Figure, *, title: str = "", dark: bool = False, height: int | None = None) -> go.Figure:
    """Aplica o template ALFA a uma figura já construída.

    Os fundos vão explícitos (e transparentes) na própria figura porque o
    Streamlit injeta a cor do tema dele no SVG mesmo com `theme=None`. Quem
    pinta o fundo é o card em volta, no CSS.
    """
    fig.update_layout(
        template="alfa_dark" if dark else "alfa",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    if title:
        fig.update_layout(title=dict(text=title))
    if height:
        fig.update_layout(height=height)
    return fig


#: Config padrão para `st.plotly_chart` — barra de ferramentas fora do caminho,
#: rolagem da página preservada no mobile.
CHART_CONFIG = {
    "displayModeBar": False,
    "scrollZoom": False,
    "responsive": True,
    "displaylogo": False,
}
