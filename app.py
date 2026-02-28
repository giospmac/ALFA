from __future__ import annotations

from ui.asset_analysis import render_asset_analysis_page
from ui.charts import render_charts_page
from ui.home import render_home_page
from ui.markowitz import render_markowitz_page
from ui.risk_analysis import render_risk_analysis_page

import streamlit as st


st.set_page_config(
    page_title="ALFA | Streamlit",
    page_icon="📊",
    layout="wide",
)


def main() -> None:
    st.sidebar.title("ALFA")
    page = st.sidebar.radio(
        "Navegação",
        options=[
            "Portfólio",
            "Gráficos",
            "Indicadores de Risco",
            "Fronteira Eficiente",
            "Análise de Ativos",
        ],
    )

    if page == "Portfólio":
        render_home_page()
        return

    if page == "Gráficos":
        render_charts_page()
        return

    if page == "Indicadores de Risco":
        render_risk_analysis_page()
        return

    if page == "Fronteira Eficiente":
        render_markowitz_page()
        return

    render_asset_analysis_page()


if __name__ == "__main__":
    main()

