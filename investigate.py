import streamlit as st
from ui.risk_analysis import render_risk_analysis_page

def main():
    try:
        render_risk_analysis_page()
    except Exception as e:
        print(f"Error caught: {e}")

if __name__ == "__main__":
    main()
