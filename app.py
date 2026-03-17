from __future__ import annotations

from ui.asset_analysis import render_asset_analysis_page
from ui.charts import render_charts_page
from ui.home import render_home_page
from ui.markowitz import render_markowitz_page
from ui.quant_projections import render_quant_projections_page
from ui.risk_analysis import render_risk_analysis_page
from ui.stock_comparison import render_stock_comparison_page
from ui.stress_scenarios import render_stress_scenarios_page

import base64
from pathlib import Path

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


def _get_base64_image(image_path: str) -> str:
    path = Path(image_path)
    if not path.is_file():
        return ""
    with open(path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode('utf-8')
    ext = path.suffix.lower().lstrip('.')
    return f"data:image/{ext};base64,{b64}"


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
            --alfa-bg-positive: #ECFDF5;
            --alfa-negative: #EF4444;
            --alfa-bg-negative: #FEF2F2;
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
            color: #4979f6;
        }
        section.main div[data-testid="stButton"] > button p,
        section.main div[data-testid="stFormSubmitButton"] > button p {
            color: #4979f6;
        }
        section.main div[data-testid="stButton"] > button[kind="primary"],
        section.main div[data-testid="stFormSubmitButton"] > button[kind="primaryFormSubmit"] {
            background: var(--alfa-accent) !important;
            border-color: var(--alfa-accent) !important;
            color: #f4f5f0 !important;
            box-shadow: 0 1px 4px rgba(37,99,235,0.25) !important;
        }
        section.main div[data-testid="stButton"] > button[kind="primary"]:hover,
        section.main div[data-testid="stFormSubmitButton"] > button[kind="primaryFormSubmit"]:hover {
            background: var(--alfa-accent-2) !important;
            border-color: var(--alfa-accent-2) !important;
        }
        section.main div[data-testid="stButton"] > button[kind="primary"] p,
        section.main div[data-testid="stFormSubmitButton"] > button[kind="primaryFormSubmit"] p {
            color: #f4f5f0 !important;
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

        /* KPI Cards (Global) */
        .alfa-kpi-card {
            background: var(--alfa-surface);
            border: 1px solid var(--alfa-border);
            border-radius: var(--alfa-radius);
            padding: 1rem 1.1rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.06);
        }
        .alfa-kpi-label {
            color: var(--alfa-muted);
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.07em;
            margin-bottom: 0.6rem;
        }
        .alfa-kpi-trend.positive {
            color: var(--alfa-positive);
            background: var(--alfa-bg-positive);
        }
        .alfa-kpi-trend.negative {
            color: var(--alfa-negative);
            background: var(--alfa-bg-negative);
        }
        .alfa-kpi-value {
            color: var(--alfa-text);
            font-size: 1.5rem;
            font-weight: 700;
            line-height: 1.15;
            margin-bottom: 0.3rem;
            letter-spacing: -0.02em;
        }
        .alfa-kpi-note {
            color: var(--alfa-accent);
            font-size: 0.82rem;
            font-weight: 500;
        }
        .alfa-section-title {
            color: #374151;
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin: 0.25rem 0 0.75rem 0;
        }

        /* Fix vertical alignment for 'Remover selecionado' button */
        div.element-container:has(.align-remove-btn) + div.element-container {
            margin-top: 28px;
        }

        /* Social Footer */
        .alfa-nav-footer {
            margin-top: auto;
            padding-top: 1.5rem;
            display: flex;
            justify-content: center;
            gap: 1.5rem;
        }
        .alfa-nav-footer a {
            color: var(--alfa-soft);
            transition: color 0.15s ease;
        }
        .alfa-nav-footer a:hover {
            color: var(--alfa-accent);
        }
        .alfa-nav-footer svg {
            width: 1.25rem;
            height: 1.25rem;
            fill: currentColor;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_navigation() -> str:
    if "current_page" not in st.session_state:
        st.session_state["current_page"] = "portfolio"

    current_page = st.session_state["current_page"]
    
    logo_path = Path(__file__).parent / "alfa_logo_blue.png"
    logo_base64 = _get_base64_image(str(logo_path))
    
    with st.sidebar:
        st.markdown(
            f"""
            <div class="alfa-nav-brand" style="text-align: center;">
                <img src="{logo_base64}" alt="ALFA Logo" style="max-width: 120px; object-fit: contain;">
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

        # Spacer
        st.markdown('<div style="flex-grow: 1; min-height: 4rem;"></div>', unsafe_allow_html=True)
        
        # Social Footer
        st.markdown(
            """
            <div class="alfa-nav-footer">
                <a href="#" target="_blank" title="LinkedIn">
                    <svg viewBox="0 0 24 24"><path d="M19 0h-14c-2.761 0-5 2.239-5 5v14c0 2.761 2.239 5 5 5h14c2.762 0 5-2.239 5-5v-14c0-2.761-2.238-5-5-5zm-11 19h-3v-11h3v11zm-1.5-12.268c-.966 0-1.75-.79-1.75-1.764s.784-1.764 1.75-1.764 1.75.79 1.75 1.764-.783 1.764-1.75 1.764zm13.5 12.268h-3v-5.604c0-3.368-4-3.113-4 0v5.604h-3v-11h3v1.765c1.396-2.586 7-2.777 7 2.476v6.759z"/></svg>
                </a>
                <a href="#" target="_blank" title="Instagram">
                    <svg viewBox="0 0 24 24"><path d="M12 2.163c3.204 0 3.584.012 4.85.07 3.252.148 4.771 1.691 4.919 4.919.058 1.265.069 1.645.069 4.849 0 3.205-.012 3.584-.069 4.849-.149 3.225-1.664 4.771-4.919 4.919-1.266.058-1.644.07-4.85.07-3.204 0-3.584-.012-4.849-.07-3.26-.149-4.771-1.699-4.919-4.92-.058-1.265-.07-1.644-.07-4.849 0-3.204.013-3.583.07-4.849.149-3.227 1.664-4.771 4.919-4.919 1.266-.057 1.645-.069 4.849-.069zm0-2.163c-3.259 0-3.667.014-4.947.072-4.358.2-6.78 2.618-6.98 6.98-.059 1.281-.073 1.689-.073 4.948 0 3.259.014 3.668.072 4.948.2 4.358 2.618 6.78 6.98 6.98 1.281.058 1.689.072 4.948.072 3.259 0 3.668-.014 4.948-.072 4.354-.2 6.782-2.618 6.979-6.98.059-1.28.073-1.689.073-4.948 0-3.259-.014-3.667-.072-4.947-.196-4.354-2.617-6.78-6.979-6.98-1.281-.059-1.69-.073-4.949-.073zm0 5.838c-3.403 0-6.162 2.759-6.162 6.162s2.759 6.163 6.162 6.163 6.162-2.759 6.162-6.163c0-3.403-2.759-6.162-6.162-6.162zm0 10.162c-2.209 0-4-1.79-4-4 0-2.209 1.791-4 4-4s4 1.791 4 4c0 2.21-1.791 4-4 4zm6.406-11.845c-.796 0-1.441.645-1.441 1.44s.645 1.44 1.441 1.44c.795 0 1.439-.645 1.439-1.44s-.644-1.44-1.439-1.44z"/></svg>
                </a>
            </div>
            """,
            unsafe_allow_html=True,
        )

    return current_page


def main() -> None:
    _render_navigation_styles()
    current_page = _render_navigation()
    PAGE_CONFIG[current_page]["render"]()


if __name__ == "__main__":
    main()
