from __future__ import annotations

from ui.asset_analysis import render_asset_analysis_page
from ui.charts import render_charts_page
from ui.home import render_home_page
from ui.markowitz import render_markowitz_page
from ui.quant_projections import render_quant_projections_page
from ui.risk_analysis import render_risk_analysis_page
from ui.stock_comparison import render_stock_comparison_page

import streamlit as st


st.set_page_config(
    page_title="ALFA | Streamlit",
    page_icon="📊",
    layout="wide",
)


PAGE_CONFIG = {
    "portfolio": {"label": "Home", "render": render_home_page},
    "charts": {"label": "Histórico", "render": render_charts_page},
    "risk": {"label": "Indicadores de Risco", "render": render_risk_analysis_page},
    "frontier": {"label": "Markowitz", "render": render_markowitz_page},
    "assets": {"label": "Análise de Ativos", "render": render_asset_analysis_page},
    "comparison": {"label": "Comparador", "render": render_stock_comparison_page},
    "quant": {"label": "Projeções Quant", "render": render_quant_projections_page},
}


def _render_navigation_styles() -> None:
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"] {
            background:
                radial-gradient(circle at top left, rgba(79, 113, 255, 0.18), transparent 32%),
                linear-gradient(180deg, #F7F9FC 0%, #EEF2F8 100%);
            border-right: 1px solid rgba(15, 23, 42, 0.08);
        }
        [data-testid="stSidebar"] .block-container {
            padding-top: 1.4rem;
            padding-bottom: 1rem;
        }
        .alfa-nav-brand {
            background: rgba(255, 255, 255, 0.72);
            border: 1px solid rgba(148, 163, 184, 0.20);
            border-radius: 18px;
            padding: 1rem 1rem 0.85rem 1rem;
            margin-bottom: 1rem;
            backdrop-filter: blur(10px);
        }
        .alfa-nav-title {
            color: #0F172A;
            font-size: 1.25rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 0.15rem;
        }
        .alfa-nav-subtitle {
            color: #64748B;
            font-size: 0.86rem;
            line-height: 1.4;
        }
        [data-testid="stSidebar"] div[data-testid="stButton"] > button {
            justify-content: flex-start;
            border-radius: 14px;
            border: 1px solid rgba(148, 163, 184, 0.18);
            background: rgba(255, 255, 255, 0.55);
            color: #0F172A;
            min-height: 2.85rem;
            font-weight: 500;
            box-shadow: none;
            transition: all 0.18s ease;
        }
        [data-testid="stSidebar"] div[data-testid="stButton"] > button:hover {
            border-color: rgba(79, 113, 255, 0.28);
            background: rgba(255, 255, 255, 0.88);
            color: #0F172A;
        }
        [data-testid="stSidebar"] div[data-testid="stButton"] > button[kind="primary"] {
            background: linear-gradient(135deg, #172554 0%, #2F5BFF 100%);
            color: #F8FAFC;
            border: 1px solid rgba(47, 91, 255, 0.38);
            box-shadow: 0 10px 24px rgba(47, 91, 255, 0.18);
        }
        [data-testid="stSidebar"] div[data-testid="stButton"] > button p {
            font-size: 0.96rem;
        }
        .alfa-nav-section {
            color: #94A3B8;
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            margin: 0.4rem 0 0.6rem 0.1rem;
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
                <div class="alfa-nav-title">ALFA</div>
                <div class="alfa-nav-subtitle">Analytics workspace para portfolio, risco e comparação de ativos.</div>
            </div>
            <div class="alfa-nav-section">Navigation</div>
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
