"""Página O Fundo — filosofia, processo de investimento e a carteira em números."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from services.portfolio_analytics import (
    benchmark_return_series,
    drawdown_series,
    individual_metrics,
    performance_indices,
    var_cvar_metrics,
)
from site_pages._shared import brl_compact, clear_fund_cache, fund_snapshot, num, pct, short_ticker
from theme import components as c
from theme import tokens as T
from theme.plotly_theme import CHART_CONFIG, style

TRADING_DAYS = 252

PROCESSO = [
    {"titulo": "Screening Setorial", "descricao": "Mapeamento de oportunidades para identificar possíveis assimetrias."},
    {"titulo": "Seleção de Empresas", "descricao": "Escolha do case mais promissor para aprofundar a análise."},
    {"titulo": "Análise Qualitativa", "descricao": "Modelo de negócio, drivers do setor e vantagens competitivas."},
    {"titulo": "Estruturação de Premissas", "descricao": "Construção do racional e das hipóteses que sustentam a modelagem."},
    {"titulo": "Análise Quantitativa", "descricao": "Modelagem e valuation por DCF e múltiplos para estimar o valor intrínseco."},
    {"titulo": "Construção da Carteira", "descricao": "Alocação dos ativos e definição de pesos, com análise de risco."},
    {"titulo": "Monitoramento", "descricao": "Acompanhamento contínuo da tese, resultados, riscos e ajustes."},
]

PILARES = [
    ("Qualidade", "Empresas com vantagens competitivas sustentáveis, boa alocação de capital e governança sólida."),
    ("Margem de segurança", "Compramos abaixo do valor intrínseco estimado — o desconto é o que protege a tese."),
    ("Horizonte longo", "Decisões pensadas em anos, não em trimestres. Giro baixo e convicção alta."),
    ("Risco medido", "Toda posição passa pelo crivo de VaR, contribuição de risco e correlação com a carteira."),
]


# ------------------------------------------------------------------ analytics


@st.cache_data(show_spinner=False, ttl=900)
def _fund_metrics(portfolio_df: pd.DataFrame, historical_df: pd.DataFrame) -> dict:
    """Indicadores da carteira na janela de 12 meses (ou no histórico disponível)."""
    empty = {"ok": False}
    if portfolio_df.empty or historical_df.empty:
        return empty

    end_date = historical_df.index.max()
    start_date = max(end_date - pd.DateOffset(years=1), historical_df.index.min())

    metrics = individual_metrics(portfolio_df, historical_df, start_date, end_date)
    if not metrics or "Portfolio" not in metrics:
        return empty

    total_pl = float(pd.to_numeric(portfolio_df["valor_real"], errors="coerce").fillna(0).sum())
    daily_vol = metrics["Portfolio"]["volatilidade"]
    daily_mean = metrics["Portfolio"]["media"]

    result = {
        "ok": True,
        "start": start_date,
        "end": end_date,
        "total_pl": total_pl,
        "asset_count": int((portfolio_df["ticker"].astype(str).str.strip() != "").sum()),
        "vol_anual": float(daily_vol * np.sqrt(TRADING_DAYS) * 100),
        "retorno_anualizado": float(((1 + daily_mean) ** TRADING_DAYS - 1) * 100),
    }

    ibov = benchmark_return_series(portfolio_df, historical_df, "IBOVESPA", years=1)
    if not ibov.empty:
        result["ret_portfolio"] = float(ibov["Portfolio"].iloc[-1])
        result["ret_ibov"] = float(ibov["IBOVESPA"].iloc[-1])
        result["curva_ibov"] = ibov

    cdi = benchmark_return_series(portfolio_df, historical_df, "CDI", years=1)
    if not cdi.empty:
        result["ret_cdi"] = float(cdi["CDI"].iloc[-1])
        result["curva_cdi"] = cdi["CDI"]

    var_metrics = var_cvar_metrics(portfolio_df, historical_df, total_pl, start_date, end_date, 0.95)
    if var_metrics and "Portfolio" in var_metrics and total_pl > 0:
        result["var_pct"] = float(var_metrics["Portfolio"]["var"] / total_pl * 100)
        result["cvar_pct"] = float(var_metrics["Portfolio"]["cvar"] / total_pl * 100)

    indices = performance_indices(portfolio_df, historical_df, start_date, end_date)
    if indices:
        result["sharpe"] = float(indices["anual"]["sharpe"])
        result["sortino"] = float(indices["anual"]["sortino"])

    drawdown = drawdown_series(portfolio_df, historical_df)
    if not drawdown.empty:
        result["max_drawdown"] = float(drawdown.min())

    return result


def _curve_chart(metrics: dict) -> go.Figure | None:
    if "curva_ibov" not in metrics:
        return None

    ibov = metrics["curva_ibov"]
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=ibov.index,
            y=ibov["Portfolio"],
            name="Carteira ALFA",
            mode="lines",
            line=dict(color=T.BLUE_500, width=2.6),
            fill="tozeroy",
            fillcolor="rgba(73,121,246,.12)",
            hovertemplate="%{y:.2f}%<extra>Carteira ALFA</extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=ibov.index,
            y=ibov["IBOVESPA"],
            name="Ibovespa",
            mode="lines",
            line=dict(color=T.ON_DARK_SOFT, width=1.8, dash="dot"),
            hovertemplate="%{y:.2f}%<extra>Ibovespa</extra>",
        )
    )
    if "curva_cdi" in metrics:
        cdi = metrics["curva_cdi"]
        figure.add_trace(
            go.Scatter(
                x=cdi.index,
                y=cdi.values,
                name="CDI",
                mode="lines",
                line=dict(color=T.POS, width=1.6),
                hovertemplate="%{y:.2f}%<extra>CDI</extra>",
            )
        )
    figure.update_yaxes(ticksuffix="%")
    return style(figure, title="Retorno acumulado · 12 meses", dark=True, height=380)


def _allocation_chart(portfolio_df: pd.DataFrame) -> go.Figure | None:
    working = portfolio_df.copy()
    working["porcentagem_real"] = pd.to_numeric(working["porcentagem_real"], errors="coerce").fillna(0.0)
    working = working[working["porcentagem_real"] > 0].sort_values("porcentagem_real")
    if working.empty:
        return None

    figure = go.Figure(
        go.Bar(
            x=working["porcentagem_real"],
            y=[short_ticker(t) for t in working["ticker"]],
            orientation="h",
            marker=dict(
                color=working["porcentagem_real"],
                colorscale=[[i / (len(T.SEQUENTIAL) - 1), color] for i, color in enumerate(T.SEQUENTIAL)],
                line=dict(width=0),
            ),
            hovertemplate="%{y}: %{x:.2f}%<extra></extra>",
        )
    )
    figure.update_xaxes(ticksuffix="%")
    figure.update_layout(hovermode="closest", showlegend=False)
    return style(figure, title="Composição da carteira", dark=True, height=max(380, 22 * len(working)))


# -------------------------------------------------------------------- blocos


def _render_header() -> None:
    c.render(
        c.section(
            c.container(
                c.reveal(c.eyebrow("ALFA Asset"), step=1)
                + c.reveal("<h1>Fundo de Investimentos</h1>", step=2)
                + c.reveal(
                    c.lead(
                        "Um fundo simulado long-only de ações, gerido pelos membros com filosofia de "
                        "<strong>Value Investing</strong>: empresas de qualidade, com vantagens competitivas "
                        "sustentáveis, negociadas abaixo do valor intrínseco."
                    ),
                    step=3,
                )
            ),
            variant="dark",
            waves="tl",
            extra="alfa-section--tight",
        )
    )


def _render_pilares() -> None:
    cards = [
        c.card(title=titulo, body=texto, step=index + 1)
        for index, (titulo, texto) in enumerate(PILARES)
    ]
    c.render(
        c.section(
            c.container(
                c.section_head(kicker="Filosofia", title="Quatro pilares que guiam cada tese", center=True)
                + c.grid(cards, cols=4)
            ),
            variant="light",
        )
    )


def _render_processo() -> None:
    c.render(
        c.section(
            c.container(
                c.section_head(
                    kicker="Processo de investimento",
                    title="Da tese à carteira, em sete etapas",
                    center=True,
                )
                + f'<div style="max-width:760px;margin:0 auto">{c.timeline(PROCESSO)}</div>',
                narrow=False,
            ),
            variant="surface",
        )
    )


def _render_vitrine(portfolio_df: pd.DataFrame, historical_df: pd.DataFrame) -> None:
    metrics = _fund_metrics(portfolio_df, historical_df)

    with st.container(key="alfaband_dark_vitrine"):
        reference = (
            f"Dados de {metrics['end'].strftime('%d/%m/%Y')}"
            if metrics.get("ok")
            else "Aguardando atualização do histórico"
        )
        c.render(
            '<div class="alfa-center">'
            + c.section_head(
                kicker="Gestão & Risco",
                title="A carteira em números",
                subtitle="Indicadores calculados pela diretoria de Gestão &amp; Risco sobre a carteira "
                "simulada — os mesmos modelos disponíveis na plataforma.",
                center=True,
            )
            + f'<p style="margin-top:-20px">{c.chip(reference)}</p></div>'
        )

        if not metrics.get("ok"):
            _render_sem_dados()
            return

        c.render(
            c.grid(
                [
                    c.kpi(
                        "Retorno 12m",
                        pct(metrics.get("ret_portfolio", float("nan")), signed=True),
                        note=f"Ibovespa {pct(metrics.get('ret_ibov', float('nan')), signed=True)}",
                        tone="up"
                        if metrics.get("ret_portfolio", 0) >= metrics.get("ret_ibov", 0)
                        else "down",
                        step=1,
                    ),
                    c.kpi("Volatilidade anual", pct(metrics["vol_anual"]), note="desvio-padrão anualizado", step=2),
                    c.kpi("Sharpe (a.a.)", num(metrics.get("sharpe", float("nan"))), note="excesso sobre o CDI", step=3),
                    c.kpi(
                        "VaR 95% diário",
                        pct(metrics.get("var_pct", float("nan"))),
                        note=f"CVaR {pct(metrics.get('cvar_pct', float('nan')))}",
                        step=4,
                    ),
                    c.kpi(
                        "Máx. drawdown",
                        pct(metrics.get("max_drawdown", float("nan"))),
                        note="pior queda no período",
                        step=5,
                    ),
                    c.kpi("Patrimônio simulado", brl_compact(metrics["total_pl"]), note=f"{metrics['asset_count']} posições", step=6),
                ],
                cols=3,
            )
        )

        st.write("")
        left, right = st.columns([1.35, 1], gap="large")
        with left:
            curve = _curve_chart(metrics)
            if curve is not None:
                st.plotly_chart(curve, use_container_width=True, theme=None, config=CHART_CONFIG)
        with right:
            allocation = _allocation_chart(portfolio_df)
            if allocation is not None:
                st.plotly_chart(allocation, use_container_width=True, theme=None, config=CHART_CONFIG)

        _render_tabela(portfolio_df)


def _render_tabela(portfolio_df: pd.DataFrame) -> None:
    working = portfolio_df.copy()
    for column in ("preco", "porcentagem_desejada", "porcentagem_real", "valor_real"):
        working[column] = pd.to_numeric(working[column], errors="coerce")
    working = working[working["ticker"].astype(str).str.strip() != ""]
    working = working.sort_values("porcentagem_real", ascending=False)
    if working.empty:
        return

    max_weight = float(working["porcentagem_real"].max() or 1)
    rows = []
    for _, row in working.iterrows():
        width = max(2.0, float(row["porcentagem_real"]) / max_weight * 100)
        rows.append(
            [
                f'<b>{c.esc(short_ticker(row["ticker"]))}</b>',
                c.esc(str(row["nome"]).title().strip()),
                f'R$ {num(row["preco"])}',
                pct(row["porcentagem_real"], decimals=2),
                f'<span class="bar" style="width:{width:.0f}%"></span>',
            ]
        )

    c.render(
        '<div style="margin-top:28px">'
        + c.reveal(
            c.data_table(
                ["Ticker", "Empresa", "Preço", "Peso", ""],
                rows,
                numeric_from=2,
            )
        )
        + "</div>"
    )


def _render_sem_dados() -> None:
    st.info(
        "O histórico consolidado ainda não cobre as posições atuais da carteira. "
        "Atualize os dados de mercado para liberar os indicadores.",
        icon=":material/update:",
    )
    with st.expander("Atualizar dados de mercado"):
        st.caption(
            "Baixa o histórico de preços das posições, dos benchmarks (Ibovespa, CDI) e dos títulos "
            "públicos. Leva de 20 a 60 segundos."
        )
        if st.button("Atualizar agora", key="fundo_refresh", type="primary"):
            from services.portfolio_analytics import build_historical_dataset
            from site_pages._shared import repository

            portfolio_df, _ = fund_snapshot()
            with st.spinner("Baixando preços e recalculando indicadores…"):
                result = build_historical_dataset(portfolio_df)
                repository().save_historical_data(result.historical)
            clear_fund_cache()
            _fund_metrics.clear()
            if result.invalid_tickers:
                st.warning("Tickers sem dados: " + ", ".join(result.invalid_tickers))
            st.rerun()


def _render_cta(on_platform) -> None:
    with st.container(key="alfaband_light_fundocta"):
        c.render(
            '<div class="alfa-center">'
            + c.section_head(
                title="Explore a matemática por trás do fundo",
                subtitle="As mesmas ferramentas de risco e otimização de carteira usadas na gestão "
                "estão abertas para qualquer pessoa usar.",
                center=True,
            )
            + "</div>"
        )
        with c.cta_row("fundoplat"):
            if st.button("Abrir a plataforma", key="fundo_cta_plataforma", type="primary"):
                on_platform()


def render(*, goto) -> None:
    portfolio_df, historical_df = fund_snapshot()
    _render_header()
    _render_pilares()
    _render_processo()
    _render_vitrine(portfolio_df, historical_df)
    _render_cta(lambda: goto("plataforma"))
