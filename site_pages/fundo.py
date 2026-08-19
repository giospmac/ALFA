"""Página O Fundo — filosofia, processo de investimento e a carteira em números."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from services.portfolio_analytics import (
    benchmark_return_series,
    capm_alpha_beta_correlation,
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

#: Data de abertura do fundo. Toda a vitrine (série do gráfico, retorno desde a
#: abertura, vol, Sharpe, VaR, alfa e drawdown) é ancorada aqui — mude só esta
#: linha se a data mudar.
ABERTURA_FUNDO = pd.Timestamp("2026-01-01")

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
    """Indicadores da carteira desde a abertura do fundo.

    Vol, Sharpe, VaR/CVaR, alfa e drawdown usam a janela desde a abertura; o
    retorno de 12 meses é o único indicador que olha para trás dela, aplicando
    os pesos atuais sobre o histórico dos ativos.
    """
    if portfolio_df.empty or historical_df.empty:
        return {"ok": False}

    fim = historical_df.index.max()
    inicio = max(ABERTURA_FUNDO, historical_df.index.min())
    if inicio >= fim:
        return {"ok": False}

    metricas = individual_metrics(portfolio_df, historical_df, inicio, fim)
    if not metricas or "Portfolio" not in metricas:
        return {"ok": False}

    total_pl = float(pd.to_numeric(portfolio_df["valor_real"], errors="coerce").fillna(0).sum())
    vol_diaria = metricas["Portfolio"]["volatilidade"]

    resultado: dict = {
        "ok": True,
        "inicio": inicio,
        "fim": fim,
        "total_pl": total_pl,
        "vol_anual": float(vol_diaria * np.sqrt(TRADING_DAYS) * 100),
    }

    # --- retorno desde a abertura (também alimenta o gráfico) -------------
    desde_abertura = benchmark_return_series(portfolio_df, historical_df, "IBOVESPA", start=inicio)
    if not desde_abertura.empty:
        resultado["ret_abertura"] = float(desde_abertura["Portfolio"].iloc[-1])
        resultado["ret_ibov_abertura"] = float(desde_abertura["IBOVESPA"].iloc[-1])
        resultado["curva"] = desde_abertura

    cdi_abertura = benchmark_return_series(portfolio_df, historical_df, "CDI", start=inicio)
    if not cdi_abertura.empty:
        resultado["curva_cdi"] = cdi_abertura["CDI"]

    # `total_pl` é o capital nocional com que o fundo abriu; o patrimônio de
    # hoje é esse capital corrigido pelo retorno acumulado desde a abertura.
    resultado["capital_inicial"] = total_pl
    resultado["pl_acumulado"] = total_pl * (1 + resultado.get("ret_abertura", 0.0) / 100)

    # --- retorno de 12 meses ---------------------------------------------
    doze_meses = benchmark_return_series(portfolio_df, historical_df, "IBOVESPA", years=1)
    if not doze_meses.empty:
        resultado["ret_12m"] = float(doze_meses["Portfolio"].iloc[-1])
        resultado["ret_ibov_12m"] = float(doze_meses["IBOVESPA"].iloc[-1])

    # --- risco ------------------------------------------------------------
    var = var_cvar_metrics(portfolio_df, historical_df, total_pl, inicio, fim, 0.95)
    if var and "Portfolio" in var and total_pl > 0:
        resultado["var_pct"] = float(var["Portfolio"]["var"] / total_pl * 100)
        resultado["cvar_pct"] = float(var["Portfolio"]["cvar"] / total_pl * 100)

    indices = performance_indices(portfolio_df, historical_df, inicio, fim)
    if indices:
        resultado["sharpe"] = float(indices["anual"]["sharpe"])

    capm = capm_alpha_beta_correlation(portfolio_df, historical_df, inicio, fim)
    if capm:
        # o alfa do OLS é diário; anualizamos para leitura na vitrine
        resultado["alfa_anual"] = float(((1 + capm["alfa"]) ** TRADING_DAYS - 1) * 100)
        resultado["beta"] = float(capm["beta"])

    drawdown = drawdown_series(portfolio_df, historical_df.loc[inicio:])
    if not drawdown.empty:
        resultado["max_drawdown"] = float(drawdown.min())

    return resultado


def _curve_chart(metrics: dict) -> go.Figure | None:
    if "curva" not in metrics:
        return None

    ibov = metrics["curva"]
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
    titulo = f"Retorno acumulado · desde {metrics['inicio'].strftime('%d/%m/%Y')}"
    return style(figure, title=titulo, dark=True, height=430)


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


def _atualizar_historico(portfolio_df: pd.DataFrame) -> None:
    """Rebaixa preços, benchmarks e títulos públicos e regrava o histórico."""
    from services.portfolio_analytics import build_historical_dataset
    from site_pages._shared import repository

    with st.spinner("Baixando preços e recalculando indicadores…"):
        resultado = build_historical_dataset(portfolio_df)
        repository().save_historical_data(resultado.historical)

    clear_fund_cache()
    _fund_metrics.clear()
    if resultado.invalid_tickers:
        st.warning("Tickers sem dados: " + ", ".join(resultado.invalid_tickers))
    st.rerun()


def _render_status(portfolio_df: pd.DataFrame, historical_df: pd.DataFrame) -> None:
    """Selo com a data-base do histórico e o botão de atualização."""
    fim = historical_df.index.max() if not historical_df.empty else None
    hoje = pd.Timestamp.today().normalize()
    # 4 dias cobrem um feriado emendado no fim de semana sem alarme falso
    defasado = fim is None or (hoje - fim).days > 4

    selo = f"Dados de {fim.strftime('%d/%m/%Y')}" if fim is not None else "Sem histórico"
    c.render(
        f'<div class="alfa-center" style="margin-top:-20px">{c.chip(selo, tone="wait" if defasado else "info")}</div>'
    )

    if defasado:
        atraso = f" — {(hoje - fim).days} dias atrás" if fim is not None else ""
        st.warning(
            f"O histórico não chega até hoje{atraso}. Atualize para recalcular os indicadores.",
            icon=":material/update:",
        )

    _, meio, _ = st.columns([1, 1.1, 1])
    with meio:
        if st.button(
            "Atualizar histórico",
            key="fundo_atualizar",
            type="primary" if defasado else "secondary",
            use_container_width=True,
        ):
            _atualizar_historico(portfolio_df)


def _cards(metrics: dict) -> list[str]:
    """Os oito indicadores da vitrine, na ordem definida pela diretoria."""
    ret_abertura = metrics.get("ret_abertura", float("nan"))
    ret_12m = metrics.get("ret_12m", float("nan"))
    alfa = metrics.get("alfa_anual", float("nan"))
    abertura = metrics["inicio"].strftime("%d/%m/%Y")

    def tom(valor: float, referencia: float) -> str:
        if pd.isna(valor) or pd.isna(referencia):
            return ""
        return "up" if valor >= referencia else "down"

    return [
        c.kpi(
            "Retorno desde a abertura",
            pct(ret_abertura, signed=True),
            note=f"Ibovespa {pct(metrics.get('ret_ibov_abertura', float('nan')), signed=True)}",
            tone=tom(ret_abertura, metrics.get("ret_ibov_abertura", float("nan"))),
            step=1,
        ),
        c.kpi(
            "Retorno 12M",
            pct(ret_12m, signed=True),
            note=f"Ibovespa {pct(metrics.get('ret_ibov_12m', float('nan')), signed=True)}",
            tone=tom(ret_12m, metrics.get("ret_ibov_12m", float("nan"))),
            step=2,
        ),
        c.kpi("Volatilidade", pct(metrics["vol_anual"]), note="desvio-padrão anualizado", step=3),
        c.kpi("Sharpe (a.a.)", num(metrics.get("sharpe", float("nan"))), note="excesso sobre o CDI", step=4),
        c.kpi(
            "VaR 95% diário",
            pct(metrics.get("var_pct", float("nan"))),
            note=f"CVaR {pct(metrics.get('cvar_pct', float('nan')))}",
            step=5,
        ),
        c.kpi(
            "Alfa (a.a.)",
            pct(alfa, decimals=2, signed=True),
            note=f"vs. Ibovespa · beta {num(metrics.get('beta', float('nan')))}",
            tone="up" if not pd.isna(alfa) and alfa >= 0 else "down",
            step=6,
        ),
        c.kpi(
            "Máx. drawdown",
            pct(metrics.get("max_drawdown", float("nan"))),
            note="pior queda desde a abertura",
            step=7,
        ),
        c.kpi(
            "Patrimônio líquido",
            brl_compact(metrics.get("pl_acumulado", metrics["total_pl"])),
            note=f"de {brl_compact(metrics['capital_inicial'])} na abertura, em {abertura}",
            step=8,
        ),
    ]


def _render_vitrine(portfolio_df: pd.DataFrame, historical_df: pd.DataFrame) -> None:
    metrics = _fund_metrics(portfolio_df, historical_df)

    with st.container(key="alfaband_dark_vitrine"):
        c.render(
            '<div class="alfa-center">'
            + c.section_head(
                kicker="Gestão & Risco",
                title="A carteira em números",
                subtitle="Indicadores calculados pela diretoria de Gestão &amp; Risco sobre a carteira "
                f"simulada, desde a abertura do fundo em {ABERTURA_FUNDO.strftime('%d/%m/%Y')}.",
                center=True,
            )
            + "</div>"
        )

        _render_status(portfolio_df, historical_df)

        if not metrics.get("ok"):
            st.info(
                "O histórico consolidado ainda não cobre as posições atuais da carteira. "
                "Use o botão acima para baixar os dados de mercado.",
                icon=":material/info:",
            )
            return

        st.write("")
        c.render(c.grid(_cards(metrics), cols=4))

        st.write("")
        curve = _curve_chart(metrics)
        if curve is not None:
            st.plotly_chart(curve, use_container_width=True, theme=None, config=CHART_CONFIG)

        _render_tabela(portfolio_df, historical_df)


def _variacao(historical_df: pd.DataFrame, ticker: str, *, meses: int) -> float | None:
    """Retorno do ativo na janela pedida, em %.

    Devolve None quando o histórico não cobre a janela inteira — melhor mostrar
    um traço do que um número calculado sobre um período mais curto.
    """
    if historical_df.empty or ticker not in historical_df.columns:
        return None
    serie = pd.to_numeric(historical_df[ticker], errors="coerce").dropna()
    if serie.empty:
        return None

    fim = serie.index.max()
    anterior = serie.loc[: fim - pd.DateOffset(months=meses)]
    if anterior.empty:
        return None

    base = float(anterior.iloc[-1])
    if base == 0:
        return None
    return (float(serie.iloc[-1]) / base - 1) * 100


def _celula_variacao(valor: float | None) -> str:
    if valor is None:
        return '<span style="color:var(--on-dark-soft)">—</span>'
    if abs(valor) < 0.005:  # arredondaria para 0,00%: sai neutro, sem sinal
        return '<span style="color:var(--on-dark-soft)">0,00%</span>'
    classe = "pos" if valor > 0 else "neg"
    return f'<span class="{classe}">{pct(valor, decimals=2, signed=True)}</span>'


def _render_tabela(portfolio_df: pd.DataFrame, historical_df: pd.DataFrame) -> None:
    working = portfolio_df.copy()
    for column in ("preco", "porcentagem_desejada", "porcentagem_real", "valor_real"):
        working[column] = pd.to_numeric(working[column], errors="coerce")
    working = working[working["ticker"].astype(str).str.strip() != ""]
    working = working.sort_values("porcentagem_real", ascending=False)
    if working.empty:
        return

    rows = []
    for _, row in working.iterrows():
        rows.append(
            [
                f'<b>{c.esc(short_ticker(row["ticker"]))}</b>',
                c.esc(str(row["nome"]).title().strip()),
                f'R$ {num(row["preco"])}',
                pct(row["porcentagem_real"], decimals=2),
                _celula_variacao(_variacao(historical_df, str(row["ticker"]), meses=1)),
                _celula_variacao(_variacao(historical_df, str(row["ticker"]), meses=12)),
            ]
        )

    c.render(
        '<div style="margin-top:28px">'
        + c.reveal(
            c.data_table(
                ["Ticker", "Empresa", "Preço", "Peso", "1 mês", "12 meses"],
                rows,
                numeric_from=2,
            )
        )
        + "</div>"
    )


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
