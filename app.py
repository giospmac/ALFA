from __future__ import annotations

from ui.asset_analysis import render_asset_analysis_page
from ui.charts import render_charts_page
from ui.home import render_home_page
from ui.markowitz import render_markowitz_page
from ui.quant_projections import render_quant_projections_page
from ui.risk_analysis import render_risk_analysis_page
from ui.stock_comparison import render_stock_comparison_page
from ui.stress_scenarios import render_stress_scenarios_page

import streamlit as st


st.set_page_config(
    page_title="ALFA",
    page_icon="https://raw.githubusercontent.com/giospmac/ALFA/refs/heads/main/icon.png",
    layout="wide",
)


PAGE_CONFIG = {
    "portfolio": {"label": "Home", "render": render_home_page},
    "charts": {"label": "Histórico", "render": render_charts_page},
    "risk": {"label": "Indicadores de Risco", "render": render_risk_analysis_page},
    "stress": {"label": "Stress Test & Cenários", "render": render_stress_scenarios_page},
    "frontier": {"label": "Markowitz", "render": render_markowitz_page},
    "assets": {"label": "Análise de Ativos", "render": render_asset_analysis_page},
    "comparison": {"label": "Comparador", "render": render_stock_comparison_page},
    "quant": {"label": "Projeções Quant", "render": render_quant_projections_page},
}


def _render_navigation_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        /* Global tokens */
        :root {
            --alfa-bg:       #F8F9FA;
            --alfa-surface:  #FFFFFF;
            --alfa-border:   #E5E7EB;
            --alfa-text:     #111827;
            --alfa-muted:    #6B7280;
            --alfa-soft:     #9CA3AF;
            --alfa-accent:   #4979f6;
            --alfa-accent-2: #2f5adf;
            --alfa-positive: #059669;
            --alfa-negative: #4979f6;
            --alfa-radius:   10px;
            --alfa-shadow:   0 1px 3px rgba(0,0,0,0.07), 0 1px 2px rgba(0,0,0,0.04);
        }

        /* Base font */
        html, body, [class*="css"], .stApp {
            font-family: 'Inter', sans-serif !important;
        }

        /* App background */
        .stApp {
            background: var(--alfa-bg) !important;
        }

        /* Sidebar */
        [data-testid="stSidebar"] {
            background: var(--alfa-surface) !important;
            border-right: 1px solid var(--alfa-border) !important;
        }
        [data-testid="stSidebar"] .block-container {
            padding-top: 1.5rem;
            padding-bottom: 1.5rem;
        }

        /* Brand block */
        .alfa-nav-brand {
            padding: 0 0 1.25rem 0;
            margin-bottom: 0.5rem;
            border-bottom: 1px solid var(--alfa-border);
        }
        .alfa-nav-logo {
            font-size: 1.1rem;
            font-weight: 700;
            letter-spacing: 0.12em;
            color: var(--alfa-text);
            text-transform: uppercase;
        }
        .alfa-nav-subtitle {
            color: var(--alfa-soft);
            font-size: 0.78rem;
            line-height: 1.5;
            margin-top: 0.2rem;
        }

        /* Section label */
        .alfa-nav-section {
            color: var(--alfa-soft);
            font-size: 0.68rem;
            font-weight: 600;
            letter-spacing: 0.1em;
            text-transform: uppercase;
            margin: 1rem 0 0.4rem 0;
        }

        /* Nav buttons */
        [data-testid="stSidebar"] div[data-testid="stButton"] > button {
            justify-content: flex-start;
            border-radius: var(--alfa-radius);
            border: 1px solid transparent;
            background: transparent;
            color: var(--alfa-muted);
            min-height: 2.6rem;
            font-weight: 500;
            font-size: 0.9rem;
            box-shadow: none;
            transition: background 0.14s ease, color 0.14s ease, border-color 0.14s ease;
        }
        [data-testid="stSidebar"] div[data-testid="stButton"] > button:hover {
            background: var(--alfa-bg);
            border-color: var(--alfa-border);
            color: var(--alfa-text);
        }
        [data-testid="stSidebar"] div[data-testid="stButton"] > button[kind="primary"] {
            background: #EFF6FF;
            color: var(--alfa-accent);
            border-color: #BFDBFE;
            font-weight: 600;
            box-shadow: none;
        }
        [data-testid="stSidebar"] div[data-testid="stButton"] > button p {
            font-size: 0.9rem;
        }

        /* Headings */
        section.main .block-container h1 {
            font-size: 1.65rem;
            font-weight: 700;
            color: var(--alfa-text);
            letter-spacing: -0.02em;
            margin-bottom: 0.15rem;
        }
        section.main .block-container h2 {
            font-size: 1.15rem;
            font-weight: 600;
            color: var(--alfa-text);
            letter-spacing: -0.01em;
        }
        section.main .block-container h3 {
            font-size: 1rem;
            font-weight: 600;
            color: var(--alfa-text);
        }
        section.main .block-container [data-testid="stCaptionContainer"] p,
        section.main .block-container .stCaption {
            color: var(--alfa-muted);
            font-size: 0.85rem;
        }

        /* Subheader */
        section.main .block-container [data-testid="StyledFullScreenFrame"] {
            border-radius: var(--alfa-radius);
        }

        /* Tables */
        [data-testid="stDataFrame"] {
            border: 1px solid var(--alfa-border) !important;
            border-radius: var(--alfa-radius) !important;
            overflow: hidden;
        }

        /* Expander */
        section.main [data-testid="stExpander"] {
            border: 1px solid var(--alfa-border) !important;
            border-radius: var(--alfa-radius) !important;
            background: var(--alfa-surface) !important;
            box-shadow: none !important;
        }

        /* Tabs */
        button[data-baseweb="tab"] {
            font-size: 0.88rem !important;
            font-weight: 500 !important;
        }

        /* Metrics */
        [data-testid="stMetric"] {
            background: var(--alfa-surface);
            border: 1px solid var(--alfa-border);
            border-radius: var(--alfa-radius);
            padding: 0.9rem 1rem;
            box-shadow: var(--alfa-shadow);
        }
        [data-testid="stMetricLabel"] p {
            color: var(--alfa-muted) !important;
            font-size: 0.8rem !important;
            font-weight: 600 !important;
            text-transform: uppercase;
            letter-spacing: 0.06em;
        }
        [data-testid="stMetricValue"] {
            color: var(--alfa-text) !important;
            font-size: 1.45rem !important;
            font-weight: 700 !important;
        }

        /* Generic buttons */
        section.main div[data-testid="stButton"] > button,
        section.main div[data-testid="stFormSubmitButton"] > button {
            border-radius: var(--alfa-radius);
            border: 1px solid var(--alfa-border);
            background: var(--alfa-surface);
            color: #4979f6;
            font-size: 0.88rem;
            font-weight: 500;
            min-height: 2.5rem;
            box-shadow: var(--alfa-shadow);
            transition: background 0.14s ease, border-color 0.14s ease;
        }
        section.main div[data-testid="stButton"] > button:hover,
        section.main div[data-testid="stFormSubmitButton"] > button:hover {
            background: #EFF6FF;
            border-color: #BFDBFE;
        }
        section.main div[data-testid="stButton"] > button[kind="primary"],
        section.main div[data-testid="stFormSubmitButton"] > button[kind="primaryFormSubmit"] {
            background: var(--alfa-accent) !important;
            border-color: var(--alfa-accent) !important;
            color: #fff !important;
            box-shadow: 0 1px 4px rgba(37,99,235,0.25) !important;
        }
        section.main div[data-testid="stButton"] > button[kind="primary"]:hover,
        section.main div[data-testid="stFormSubmitButton"] > button[kind="primaryFormSubmit"]:hover {
            background: var(--alfa-accent-2) !important;
            border-color: var(--alfa-accent-2) !important;
        }
        section.main div[data-testid="stButton"] > button[kind="primary"] p,
        section.main div[data-testid="stFormSubmitButton"] > button[kind="primaryFormSubmit"] p {
            color: #fff !important;
        }

        /* Inputs */
        [data-baseweb="base-input"] {
            border-radius: var(--alfa-radius) !important;
            border: 1px solid var(--alfa-border) !important;
            background: var(--alfa-surface) !important;
            box-shadow: none !important;
        }
        [data-baseweb="base-input"]:focus-within {
            border-color: #93C5FD !important;
            box-shadow: 0 0 0 3px rgba(59,130,246,0.1) !important;
        }
        input, select, textarea {
            font-family: 'Inter', sans-serif !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_navigation() -> str:
    if "current_page" not in st.session_state:
        st.session_state["current_page"] = "portfolio"

    current_page = st.session_state["current_page"]
    with st.sidebar:
        st.markdown(
            """
            <div class="alfa-nav-brand">
                <div class="alfa-nav-logo">ALFA</div>
                <div class="alfa-nav-subtitle">Portfolio, risco e análise de ativos.</div>
            </div>
            <div class="alfa-nav-section">Navegação</div>
            """,
            unsafe_allow_html=True,
        )

        for page_key, page_meta in PAGE_CONFIG.items():
            is_selected = current_page == page_key
            if st.button(
                page_meta["label"],
                key=f"nav-{page_key}",
                use_container_width=True,
                type="primary" if is_selected else "secondary",
            ):
                if current_page == page_key:
                    continue
                st.session_state["current_page"] = page_key
                st.rerun()

    return current_page


def main() -> None:
    _render_navigation_styles()
    current_page = _render_navigation()
    PAGE_CONFIG[current_page]["render"]()


if __name__ == "__main__":
    main()
