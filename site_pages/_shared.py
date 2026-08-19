"""Utilidades compartilhadas pelas páginas institucionais."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from core.portfolio_repository import PortfolioRepository, PortfolioSnapshot

ROOT = Path(__file__).resolve().parents[1]
CONTENT_DIR = ROOT / "content"
MEMBER_PHOTOS = ROOT / "assets" / "membros"
ACTIVITY_PHOTOS = ROOT / "assets" / "atividades"


# ------------------------------------------------------------------ conteúdo


@st.cache_data(show_spinner=False)
def load_content(name: str) -> dict[str, Any]:
    """Lê `content/<name>.json`. Editar o JSON é suficiente para mudar o site."""
    path = CONTENT_DIR / f"{name}.json"
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:  # conteúdo editado à mão pode quebrar
        st.error(f"`content/{name}.json` está com erro de formatação (linha {exc.lineno}).")
        return {}


@st.cache_data(show_spinner=False)
def photo_uri(folder: str, filename: str) -> str:
    """Data URI de uma foto em `assets/<folder>/`. Vazio se não existir."""
    if not filename:
        return ""
    import base64

    path = ROOT / "assets" / folder / filename
    if not path.is_file():
        return ""
    mime = {"png": "image/png", "webp": "image/webp"}.get(path.suffix.lower().lstrip("."), "image/jpeg")
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"


# ------------------------------------------------------------- dados do fundo


def repository() -> PortfolioRepository:
    return PortfolioRepository(base_path=ROOT)


@st.cache_data(show_spinner=False, ttl=900)
def fund_snapshot() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carteira e histórico salvos em disco (mesmos CSVs usados pela plataforma)."""
    snapshot: PortfolioSnapshot = repository().load_snapshot()
    return snapshot.portfolio, snapshot.historical


def clear_fund_cache() -> None:
    fund_snapshot.clear()


# ------------------------------------------------------------------ formatação


def brl(value: float, *, decimals: int = 2) -> str:
    """Formata em Real no padrão brasileiro (1.234.567,89)."""
    if value is None or pd.isna(value):
        return "—"
    text = f"{value:,.{decimals}f}"
    return "R$ " + text.replace(",", "\x00").replace(".", ",").replace("\x00", ".")


def brl_compact(value: float) -> str:
    """R$ 12,4 mi / R$ 1,2 bi — para KPIs onde o número inteiro polui."""
    if value is None or pd.isna(value):
        return "—"
    absolute = abs(value)
    for threshold, suffix in ((1e9, " bi"), (1e6, " mi"), (1e3, " mil")):
        if absolute >= threshold:
            return f"R$ {value / threshold:,.1f}".replace(".", ",") + suffix
    return brl(value)


def pct(value: float, *, decimals: int = 1, signed: bool = False) -> str:
    if value is None or pd.isna(value):
        return "—"
    text = f"{value:,.{decimals}f}".replace(".", ",")
    if signed and value > 0:
        text = "+" + text
    return f"{text}%"


def num(value: float, *, decimals: int = 2) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{value:,.{decimals}f}".replace(",", "\x00").replace(".", ",").replace("\x00", ".")


def short_ticker(ticker: str) -> str:
    return str(ticker).replace(".SA", "")
