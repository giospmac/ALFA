"""Tokens da identidade visual ALFA.

Fonte: `styles.md` do app Streamlit + `css/tokens.css` do site institucional.
Estes valores são a ÚNICA fonte de verdade de cor do projeto: o CSS
(`theme/styles.py`) e os gráficos Plotly (`theme/plotly_theme.py`) leem daqui.
"""

from __future__ import annotations

# --- Base escura (heros, seções institucionais, rodapé) -------------------
NAVY_950 = "#0a1128"
NAVY_900 = "#101a3a"
NAVY_800 = "#1a2650"

# --- Azul ALFA ------------------------------------------------------------
BLUE_500 = "#4979f6"   # primária: CTAs, links, série principal
BLUE_600 = "#2f5adf"   # hover da primária
BLUE_300 = "#8aa8fa"   # apoio sobre fundo escuro, séries secundárias
BLUE_100 = "#dbe5fd"   # fundos sutis de destaque

# --- Base clara -----------------------------------------------------------
BG = "#f2f1ec"         # off-white das seções claras
SURFACE = "#ffffff"
BORDER = "#e2e0d8"
INK = "#14192e"
INK_SOFT = "#5a6178"

# --- Texto sobre fundo escuro --------------------------------------------
ON_DARK = "#f5f6f8"
ON_DARK_SOFT = "#9fb0d8"

# --- Semânticas financeiras ----------------------------------------------
POS = "#2e9e6b"
NEG = "#d64550"
WARN = "#c9922e"

# --- Paleta categórica para gráficos -------------------------------------
# Ordenada para máximo contraste entre séries vizinhas mantendo o azul ALFA
# como cor principal.
CATEGORICAL = [
    BLUE_500,
    "#0f9d8f",
    "#8a5cf0",
    "#e08a2e",
    "#2e9e6b",
    "#d64550",
    "#3d6ec9",
    "#a0a8bd",
    "#6f42c1",
    "#1a7f9c",
]

# Escala sequencial azul (mapas de calor, gradientes de peso)
SEQUENTIAL = ["#eef3fe", "#dbe5fd", "#b3c8fb", "#8aa8fa", "#4979f6", "#2f5adf", "#1d3a9c"]

# Escala divergente (correlações: -1 vermelho / 0 neutro / +1 azul)
DIVERGING = ["#d64550", "#e8a1a5", "#f2f1ec", "#8aa8fa", "#2f5adf"]

FONT_STACK = '"Inter", system-ui, -apple-system, "Segoe UI", sans-serif'
