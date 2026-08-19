"""CSS global do site ALFA.

Toda a aparência do app vive aqui. A folha é injetada uma única vez por
execução (`inject_global_css`) e é escrita em cima dos tokens de
`theme/tokens.py`, que continuam sendo a fonte de verdade das cores.

Organização:
  1. tokens             6. hero + animações
  2. reset do Streamlit 7. componentes institucionais
  3. tipografia         8. re-skin dos widgets do Streamlit
  4. layout / seções    9. responsivo (mobile-first nos breakpoints)
  5. topbar fixa       10. rodapé
"""

from __future__ import annotations

import base64
from functools import lru_cache
from pathlib import Path

import streamlit as st

from theme import tokens as T

ASSETS = Path(__file__).resolve().parents[1] / "assets"

# Altura da barra fixa (usada também pelo hero para deslizar por baixo dela).
NAV_H = 76
NAV_H_MOBILE = 64


@lru_cache(maxsize=32)
def asset_data_uri(filename: str) -> str:
    """Devolve um asset local como data URI (o Streamlit não serve /assets)."""
    path = ASSETS / filename
    if not path.is_file():
        return ""
    mime = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".svg": "image/svg+xml",
        ".webp": "image/webp",
    }.get(path.suffix.lower(), "application/octet-stream")
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"


def _css() -> str:
    waves = asset_data_uri("waves.svg")
    return f"""
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

/* ====================================================================
   1. TOKENS
   ==================================================================== */
:root {{
  --navy-950:{T.NAVY_950}; --navy-900:{T.NAVY_900}; --navy-800:{T.NAVY_800};
  --blue-500:{T.BLUE_500}; --blue-600:{T.BLUE_600};
  --blue-300:{T.BLUE_300}; --blue-100:{T.BLUE_100};
  --bg:{T.BG}; --surface:{T.SURFACE}; --border:{T.BORDER};
  --ink:{T.INK}; --ink-soft:{T.INK_SOFT};
  --on-dark:{T.ON_DARK}; --on-dark-soft:{T.ON_DARK_SOFT};
  --pos:{T.POS}; --neg:{T.NEG}; --warn:{T.WARN};

  /* Troque --font-display por uma serifada (ex.: "Newsreader", serif) se um
     dia a identidade do ALFA passar a admitir display serifado. */
  --font: {T.FONT_STACK};
  --font-display: var(--font);

  --radius: 14px;
  --radius-lg: 22px;
  --radius-pill: 999px;
  --maxw: 1180px;
  --nav-h: {NAV_H}px;

  --shadow-sm: 0 1px 2px rgba(10,17,40,.06), 0 1px 3px rgba(10,17,40,.05);
  --shadow-md: 0 6px 24px rgba(10,17,40,.08), 0 2px 6px rgba(10,17,40,.04);
  --shadow-lg: 0 24px 60px rgba(10,17,40,.14), 0 6px 16px rgba(10,17,40,.06);
  --shadow-glow: 0 18px 50px rgba(73,121,246,.28);

  --ease: cubic-bezier(.22,.61,.36,1);
}}

/* ====================================================================
   2. RESET DO STREAMLIT
   ==================================================================== */
[data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"],
[data-testid="stStatusWidget"], #MainMenu {{ display: none !important; }}
/* NB: não esconder `footer` genérico — o rodapé do site é um <footer>. */
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"] {{ display: none !important; }}

html, body, .stApp {{
  font-family: var(--font) !important;
  background: var(--bg) !important;
  color: var(--ink);
  -webkit-font-smoothing: antialiased;
  overflow-x: hidden;
}}
html {{ scroll-behavior: smooth; }}

/* O container principal vira "tela cheia": cada seção controla o próprio
   respiro. É isso que permite faixas full-bleed alternando claro/escuro. */
[data-testid="stMainBlockContainer"] {{
  padding: var(--nav-h) 0 0 0 !important;   /* compensa a topbar fixa */
  max-width: 100% !important;
}}
[data-testid="stMain"] {{ background: var(--bg); }}

/* Espaçamento vertical padrão entre elementos, mais generoso que o default */
[data-testid="stVerticalBlock"] {{ gap: .85rem; }}

::selection {{ background: var(--blue-500); color: #fff; }}

::-webkit-scrollbar {{ width: 10px; height: 10px; }}
::-webkit-scrollbar-track {{ background: transparent; }}
::-webkit-scrollbar-thumb {{ background: rgba(90,97,120,.28); border-radius: 999px; }}
::-webkit-scrollbar-thumb:hover {{ background: rgba(90,97,120,.45); }}

/* ====================================================================
   3. TIPOGRAFIA
   ==================================================================== */
h1, h2, h3, h4 {{ font-family: var(--font-display); line-height: 1.1; margin: 0 0 .6rem; color: var(--ink); }}
h1 {{ font-size: clamp(2.1rem, 5.2vw, 3.5rem); font-weight: 800; letter-spacing: -.035em; }}
h2 {{ font-size: clamp(1.6rem, 3.6vw, 2.4rem); font-weight: 800; letter-spacing: -.03em; }}
h3 {{ font-size: clamp(1.05rem, 2vw, 1.25rem); font-weight: 700; letter-spacing: -.015em; }}
p  {{ margin: 0 0 1rem; line-height: 1.65; }}
a  {{ color: var(--blue-500); text-decoration: none; transition: color .15s var(--ease); }}
a:hover {{ color: var(--blue-600); }}

.alfa-eyebrow {{
  display: inline-flex; align-items: center; gap: .5rem;
  font-size: .74rem; font-weight: 700; letter-spacing: .16em; text-transform: uppercase;
  color: var(--blue-500); margin-bottom: .9rem;
}}
.alfa-eyebrow::before {{
  content: ""; width: 22px; height: 1.5px; background: currentColor; opacity: .55;
}}
.alfa-lead {{ font-size: clamp(1rem, 1.6vw, 1.16rem); color: var(--ink-soft); line-height: 1.7; max-width: 46rem; }}
.alfa-muted {{ color: var(--ink-soft); }}
.alfa-center {{ text-align: center; }}
.alfa-center .alfa-lead {{ margin-left: auto; margin-right: auto; }}
.alfa-center .alfa-eyebrow::before {{ display: none; }}

/* ====================================================================
   4. LAYOUT / SEÇÕES FULL-BLEED
   ==================================================================== */
.alfa-section {{
  width: 100%;
  padding: clamp(56px, 8vw, 104px) 0;
  position: relative;
  overflow: hidden;
}}
.alfa-section--light {{ background: var(--bg); color: var(--ink); }}
.alfa-section--surface {{ background: var(--surface); color: var(--ink); }}
.alfa-section--dark {{ background: var(--navy-950); color: var(--on-dark); }}
.alfa-section--tight {{ padding: clamp(40px, 5vw, 64px) 0; }}

.alfa-section--dark h1, .alfa-section--dark h2, .alfa-section--dark h3 {{ color: var(--on-dark); }}
.alfa-section--dark p, .alfa-section--dark .alfa-lead, .alfa-section--dark .alfa-muted {{ color: var(--on-dark-soft); }}
.alfa-section--dark .alfa-eyebrow {{ color: var(--blue-300); }}

/* Emenda suave entre uma faixa escura e a clara seguinte (efeito "fade") */
.alfa-section--dark::after {{
  content: ""; position: absolute; left: 0; right: 0; bottom: 0; height: 120px;
  background: linear-gradient(to bottom, transparent, rgba(242,241,236,.10));
  pointer-events: none;
}}

.alfa-container {{
  max-width: var(--maxw);
  margin: 0 auto;
  padding: 0 clamp(20px, 4vw, 32px);
  position: relative;
  z-index: 1;
}}
.alfa-container--narrow {{ max-width: 800px; }}

.alfa-grid {{ display: grid; gap: clamp(16px, 2vw, 24px); }}
.alfa-grid--2 {{ grid-template-columns: repeat(auto-fit, minmax(215px, 1fr)); }}
.alfa-grid--3 {{ grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); }}
.alfa-grid--4 {{ grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); }}
.alfa-split {{ display: grid; grid-template-columns: 1fr 1fr; gap: clamp(24px, 4vw, 56px); align-items: center; }}

/* Ondas decorativas da identidade */
.alfa-waves {{
  position: absolute; inset: -10% -10% auto -10%; height: 130%;
  background-image: url("{waves}");
  background-size: cover; background-position: center;
  opacity: .35; pointer-events: none;
}}
.alfa-waves--br {{ transform: rotate(180deg); }}

/* ====================================================================
   5. TOPBAR FIXA (marca + navegação)
   ==================================================================== */
.st-key-alfa_topnav {{
  position: fixed; top: 0; left: 0; right: 0; z-index: 1000;
  height: var(--nav-h);
  background: rgba(10,17,40,.82);
  backdrop-filter: blur(16px) saturate(140%);
  -webkit-backdrop-filter: blur(16px) saturate(140%);
  border-bottom: 1px solid rgba(138,168,250,.16);
  flex-direction: row !important; align-items: center; flex-wrap: nowrap;
  gap: clamp(16px, 3.5vw, 44px);
  padding: 0 clamp(14px, 4vw, 32px);
}}
.st-key-alfa_topnav [data-testid="stElementContainer"] {{ margin: 0 !important; }}
/* marca: largura intrínseca, não pode "comer" a linha inteira do flex */
.st-key-alfa_topnav > [data-testid="stElementContainer"]:first-child,
.st-key-alfa_topnav > [data-testid="stElementContainer"]:first-child [data-testid="stMarkdown"],
.st-key-alfa_topnav > [data-testid="stElementContainer"]:first-child [data-testid="stMarkdownContainer"],
.st-key-alfa_topnav .alfa-brand {{ flex: 0 0 auto !important; width: max-content !important; }}
/* navegação: ocupa o resto e rola na horizontal quando não couber */
.st-key-alfa_topnav > .st-key-alfa_nav_pills {{
  flex: 1 1 auto !important; min-width: 0 !important; width: auto !important;
  overflow-x: auto; overflow-y: hidden; scrollbar-width: none;
}}
.st-key-alfa_topnav > .st-key-alfa_nav_pills::-webkit-scrollbar {{ display: none; }}
.st-key-alfa_topnav [data-testid="stWidgetLabel"] {{ display: none !important; }}

.alfa-brand {{ display: flex; align-items: center; gap: 10px; }}
.alfa-brand img {{ height: 26px; width: auto; display: block; }}
.alfa-brand-txt {{ color: var(--on-dark); font-weight: 800; letter-spacing: .04em; font-size: .95rem; line-height: 1; }}
.alfa-brand-sub {{ color: var(--on-dark-soft); font-size: .66rem; letter-spacing: .1em; text-transform: uppercase; }}

/* ====================================================================
   6. HERO + ANIMAÇÕES
   ==================================================================== */
.st-key-alfa_hero {{
  margin-top: calc(var(--nav-h) * -1);
  padding-top: calc(var(--nav-h) + clamp(56px, 9vw, 104px));
  padding-bottom: clamp(64px, 9vw, 110px);
  background: var(--navy-950);
  color: var(--on-dark);
  min-height: min(760px, 92vh);
  display: flex; align-items: center;
}}
.st-key-alfa_hero::before {{
  content: ""; position: absolute; inset: -30% -20% auto -20%; height: 150%;
  background:
    radial-gradient(46% 46% at 22% 18%, rgba(73,121,246,.34), transparent 68%),
    radial-gradient(42% 42% at 80% 26%, rgba(138,168,250,.22), transparent 66%),
    radial-gradient(52% 52% at 52% 96%, rgba(47,90,223,.30), transparent 70%);
  filter: blur(6px);
  animation: alfaAurora 22s var(--ease) infinite alternate;
  pointer-events: none;
}}
@keyframes alfaAurora {{
  0%   {{ transform: translate3d(0,0,0) scale(1); }}
  50%  {{ transform: translate3d(-3%,2%,0) scale(1.08); }}
  100% {{ transform: translate3d(3%,-2%,0) scale(1.03); }}
}}
/* Malha sutil de grid, referência "quant" */
.st-key-alfa_hero::after {{
  content: ""; position: absolute; inset: 0;
  background-image:
    linear-gradient(rgba(138,168,250,.055) 1px, transparent 1px),
    linear-gradient(90deg, rgba(138,168,250,.055) 1px, transparent 1px);
  background-size: 64px 64px;
  mask-image: radial-gradient(72% 62% at 50% 42%, #000 30%, transparent 100%);
  -webkit-mask-image: radial-gradient(72% 62% at 50% 42%, #000 30%, transparent 100%);
  pointer-events: none;
}}
.alfa-hero-logo {{ height: clamp(46px, 7vw, 68px); margin: 0 auto clamp(18px, 2.6vw, 26px); display: block; }}
.st-key-alfa_hero h1 {{ color: #fff; max-width: 15ch; margin-inline: auto; }}
.st-key-alfa_hero .alfa-lead {{ color: var(--on-dark-soft); font-size: clamp(1rem, 1.8vw, 1.2rem); margin-inline: auto; max-width: 42rem; }}

/* A 1.6x aplica `text-align: left` no stElementContainer, anulando a
   centralização herdada do hero. Reforçamos nos descendentes. */
.st-key-alfa_hero div, .st-key-alfa_hero h1,
.st-key-alfa_hero p, .st-key-alfa_hero span {{ text-align: center; }}

/* Lockup do hero: marca, nome e assinatura */
.alfa-hero-title {{
  color: #fff; font-weight: 900; line-height: 1;
  font-size: clamp(3rem, 9vw, 5.4rem); letter-spacing: .02em;
  margin: 0 0 clamp(14px, 2vw, 20px); max-width: none;
}}
.alfa-hero-sub {{
  color: var(--blue-300); font-weight: 600;
  font-size: clamp(1rem, 2.2vw, 1.35rem); letter-spacing: -.005em;
  line-height: 1.45; margin: 0 auto clamp(18px, 2.6vw, 26px); max-width: 32rem;
}}

.alfa-scrollcue {{
  margin-top: clamp(30px, 5vw, 52px);
  color: var(--on-dark-soft); font-size: .68rem; letter-spacing: .22em; text-transform: uppercase;
  display: flex; flex-direction: column; align-items: center; gap: 10px;
}}
.alfa-scrollcue span.line {{
  width: 1px; height: 34px;
  background: linear-gradient(var(--blue-300), transparent);
  animation: alfaCue 2.2s ease-in-out infinite;
}}
@keyframes alfaCue {{ 0%,100% {{ opacity:.25; transform: scaleY(.6); }} 50% {{ opacity:1; transform: scaleY(1); }} }}

@keyframes alfaRise {{ from {{ opacity: 0; transform: translateY(22px); }} to {{ opacity: 1; transform: none; }} }}
@keyframes alfaFade {{ from {{ opacity: 0; }} to {{ opacity: 1; }} }}
.alfa-reveal {{ animation: alfaRise .8s var(--ease) both; }}
.alfa-reveal-1 {{ animation-delay: .05s; }}
.alfa-reveal-2 {{ animation-delay: .13s; }}
.alfa-reveal-3 {{ animation-delay: .21s; }}
.alfa-reveal-4 {{ animation-delay: .29s; }}
.alfa-reveal-5 {{ animation-delay: .37s; }}
.alfa-reveal-6 {{ animation-delay: .45s; }}

@media (prefers-reduced-motion: reduce) {{
  *, *::before, *::after {{ animation: none !important; transition: none !important; }}
}}

/* ====================================================================
   7. COMPONENTES INSTITUCIONAIS
   ==================================================================== */
.alfa-card {{
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: clamp(18px, 2.4vw, 26px);
  box-shadow: var(--shadow-sm);
  transition: transform .28s var(--ease), box-shadow .28s var(--ease), border-color .28s var(--ease);
  height: 100%;
}}
.alfa-card:hover {{ transform: translateY(-4px); box-shadow: var(--shadow-md); border-color: var(--blue-300); }}
.alfa-card h3 {{ color: var(--blue-600); }}
.alfa-card p {{ color: var(--ink-soft); font-size: .94rem; margin: 0; }}
.alfa-card__tag {{
  display: inline-block; font-size: .64rem; font-weight: 700; letter-spacing: .13em;
  text-transform: uppercase; color: var(--ink-soft); margin-bottom: .55rem;
}}
.alfa-section--dark .alfa-card {{
  background: linear-gradient(160deg, rgba(26,38,80,.72), rgba(16,26,58,.86));
  border-color: rgba(138,168,250,.16); box-shadow: none;
}}
.alfa-section--dark .alfa-card:hover {{ border-color: rgba(138,168,250,.42); box-shadow: 0 18px 44px rgba(0,0,0,.34); }}
.alfa-section--dark .alfa-card h3 {{ color: var(--blue-300); }}
.alfa-section--dark .alfa-card p, .alfa-section--dark .alfa-card__tag {{ color: var(--on-dark-soft); }}

/* Números grandes (highlights) */
.alfa-stat {{ text-align: center; padding: clamp(14px, 2vw, 22px) 8px; }}
.alfa-stat__value {{
  font-family: var(--font-display);
  font-size: clamp(2.2rem, 5.4vw, 3.3rem); font-weight: 900; line-height: 1;
  letter-spacing: -.045em;
  background: linear-gradient(135deg, #ffffff 12%, var(--blue-300) 96%);
  -webkit-background-clip: text; background-clip: text; color: transparent;
}}
.alfa-section--light .alfa-stat__value, .alfa-section--surface .alfa-stat__value {{
  background: linear-gradient(135deg, var(--blue-600) 8%, var(--blue-500) 92%);
  -webkit-background-clip: text; background-clip: text; color: transparent;
}}
.alfa-stat__label {{ color: var(--on-dark-soft); font-size: .88rem; margin-top: .5rem; font-weight: 500; }}
.alfa-section--light .alfa-stat__label, .alfa-section--surface .alfa-stat__label {{ color: var(--ink-soft); }}

/* Pílulas / tags */
.alfa-tag {{
  display: inline-block; padding: 7px 17px; margin: 0 8px 8px 0;
  border: 1.4px solid var(--blue-500); border-radius: var(--radius-pill);
  color: var(--blue-500); font-weight: 500; font-size: .87rem;
  transition: background .2s var(--ease), color .2s var(--ease), transform .2s var(--ease);
}}
.alfa-tag:hover {{ background: var(--blue-500); color: #fff; transform: translateY(-2px); }}
.alfa-section--dark .alfa-tag {{ border-color: var(--blue-300); color: var(--blue-300); }}
.alfa-section--dark .alfa-tag:hover {{ background: var(--blue-300); color: var(--navy-950); }}

.alfa-chip {{
  display: inline-block; padding: 4px 12px; border-radius: var(--radius-pill);
  font-size: .7rem; font-weight: 700; letter-spacing: .07em; text-transform: uppercase;
}}
.alfa-chip--ok {{ background: rgba(46,158,107,.13); color: var(--pos); }}
.alfa-chip--wait {{ background: rgba(201,146,46,.15); color: var(--warn); }}
.alfa-chip--info {{ background: var(--blue-100); color: var(--blue-600); }}
.alfa-section--dark .alfa-chip--info {{ background: rgba(138,168,250,.16); color: var(--blue-300); }}

/* Botões-link (CTA em HTML) */
.alfa-btn {{
  display: inline-flex; align-items: center; gap: .55rem;
  padding: 13px 28px; border-radius: var(--radius-pill);
  font-weight: 600; font-size: .96rem; border: 1.5px solid transparent;
  transition: transform .2s var(--ease), box-shadow .2s var(--ease), background .2s var(--ease);
}}
.alfa-btn--primary {{ background: linear-gradient(135deg, var(--blue-500), var(--blue-600)); color: #fff !important; box-shadow: var(--shadow-glow); }}
.alfa-btn--primary:hover {{ transform: translateY(-2px); box-shadow: 0 22px 56px rgba(73,121,246,.38); color: #fff !important; }}
.alfa-btn--ghost {{ border-color: rgba(138,168,250,.55); color: var(--blue-300) !important; }}
.alfa-btn--ghost:hover {{ background: rgba(138,168,250,.12); transform: translateY(-2px); }}
.alfa-ctas {{ display: flex; gap: 14px; flex-wrap: wrap; justify-content: center; margin-top: 6px; }}

/* Timeline (processo do fundo / jornada do membro) */
.alfa-timeline {{ list-style: none !important; margin: 0 !important; padding: 0 !important; position: relative; }}
.alfa-timeline::before {{
  content: ""; position: absolute; left: 21px; top: 10px; bottom: 10px; width: 2px;
  background: linear-gradient(var(--blue-500), rgba(73,121,246,.08));
}}
/* !important: o CSS de markdown do Streamlit injeta padding/margin em <li>. */
.alfa-tl-step {{
  position: relative; list-style: none !important;
  margin: 0 !important; padding: 0 0 clamp(24px, 3vw, 34px) 66px !important;
}}
.alfa-tl-step:last-child {{ padding-bottom: 0 !important; }}
.alfa-tl-step::before {{
  content: attr(data-step); position: absolute; left: 0; top: -2px;
  width: 44px; height: 44px; border-radius: 50%;
  background: var(--surface); color: var(--blue-600);
  border: 2px solid var(--blue-500);
  display: flex; align-items: center; justify-content: center;
  font-weight: 800; font-size: .95rem;
  box-shadow: 0 0 0 6px rgba(73,121,246,.08);
  transition: transform .25s var(--ease), box-shadow .25s var(--ease);
}}
.alfa-tl-step:hover::before {{ transform: scale(1.08); box-shadow: 0 0 0 10px rgba(73,121,246,.13); }}
.alfa-tl-step h3 {{ margin-bottom: .25rem; }}
.alfa-tl-step p {{ color: var(--ink-soft); margin: 0; font-size: .95rem; }}
.alfa-section--dark .alfa-tl-step::before {{ background: var(--navy-900); color: var(--blue-300); }}
.alfa-section--dark .alfa-tl-step p {{ color: var(--on-dark-soft); }}

/* Estado vazio (páginas ainda sem conteúdo publicado) */
.alfa-empty {{
  max-width: 560px; margin: 0 auto; text-align: center;
  background: var(--surface); border: 1px dashed var(--border);
  border-radius: var(--radius-lg); padding: clamp(34px, 5vw, 54px) clamp(22px, 4vw, 38px);
}}
.alfa-empty h3 {{ margin-bottom: .5rem; }}
.alfa-empty p {{ color: var(--ink-soft); margin: 0; font-size: .95rem; }}
[class*="st-key-alfaband_dark"] .alfa-empty,
.alfa-section--dark .alfa-empty {{
  background: rgba(16,26,58,.6); border-color: rgba(138,168,250,.24);
}}
.alfa-section--dark .alfa-empty p {{ color: var(--on-dark-soft); }}

/* Membros */
.alfa-member {{ text-align: center; }}
.alfa-member__avatar, .alfa-member img {{
  width: 108px; height: 108px; border-radius: 50%; object-fit: cover;
  margin: 0 auto .9rem; display: flex; align-items: center; justify-content: center;
  background: linear-gradient(135deg, var(--blue-100), #eef3fe);
  color: var(--blue-600); font-size: 1.75rem; font-weight: 800; letter-spacing: -.02em;
  border: 2px solid rgba(73,121,246,.18);
  transition: transform .3s var(--ease), border-color .3s var(--ease);
}}
.alfa-member:hover .alfa-member__avatar, .alfa-member:hover img {{ transform: scale(1.05); border-color: var(--blue-500); }}
.alfa-member__name {{ font-weight: 700; font-size: 1rem; color: var(--ink); margin-bottom: 2px; }}
.alfa-member__role {{ color: var(--ink-soft); font-size: .86rem; }}

/* Agenda / calendário */
.alfa-agenda {{ display: flex; flex-direction: column; }}
.alfa-agenda__row {{
  display: flex; gap: 18px; align-items: baseline;
  padding: 13px 4px; border-bottom: 1px solid rgba(138,168,250,.14);
  transition: background .2s var(--ease), padding-left .2s var(--ease);
}}
.alfa-agenda__row:last-child {{ border-bottom: 0; }}
.alfa-agenda__row:hover {{ background: rgba(138,168,250,.06); padding-left: 12px; }}
.alfa-agenda__date {{
  flex: 0 0 62px; font-variant-numeric: tabular-nums; font-weight: 700;
  color: var(--blue-300); font-size: .9rem;
}}
.alfa-agenda__title {{ color: var(--on-dark-soft); font-size: .95rem; }}
.alfa-agenda__title--hl {{ color: var(--on-dark); font-weight: 600; }}

/* Faixa de cotações (marquee) */
.alfa-ticker {{
  overflow: hidden; border-top: 1px solid rgba(138,168,250,.14);
  border-bottom: 1px solid rgba(138,168,250,.14);
  background: rgba(10,17,40,.55); padding: 11px 0;
}}
.alfa-ticker__track {{
  display: inline-flex; gap: 34px; white-space: nowrap;
  animation: alfaMarquee 42s linear infinite;
}}
.alfa-ticker:hover .alfa-ticker__track {{ animation-play-state: paused; }}
@keyframes alfaMarquee {{ from {{ transform: translateX(0); }} to {{ transform: translateX(-50%); }} }}
.alfa-ticker__item {{ font-size: .82rem; font-variant-numeric: tabular-nums; color: var(--on-dark-soft); }}
.alfa-ticker__item b {{ color: var(--on-dark); font-weight: 700; margin-right: 7px; letter-spacing: .02em; }}
.alfa-ticker__item .up {{ color: #5fd39b; }}
.alfa-ticker__item .down {{ color: #ff8f97; }}

/* KPI em HTML (usado nas vitrines) */
.alfa-kpi {{
  background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius);
  padding: 18px 20px; box-shadow: var(--shadow-sm); height: 100%;
  transition: transform .25s var(--ease), box-shadow .25s var(--ease);
}}
.alfa-kpi:hover {{ transform: translateY(-3px); box-shadow: var(--shadow-md); }}
.alfa-kpi__label {{
  color: var(--ink-soft); font-size: .7rem; font-weight: 700;
  letter-spacing: .1em; text-transform: uppercase; margin-bottom: .55rem;
}}
.alfa-kpi__value {{
  font-size: clamp(1.35rem, 2.4vw, 1.7rem); font-weight: 800; letter-spacing: -.03em;
  color: var(--ink); line-height: 1.1; font-variant-numeric: tabular-nums;
}}
.alfa-kpi__note {{ font-size: .8rem; color: var(--ink-soft); margin-top: .35rem; }}
.alfa-kpi__note.up {{ color: var(--pos); font-weight: 600; }}
.alfa-kpi__note.down {{ color: var(--neg); font-weight: 600; }}
.alfa-section--dark .alfa-kpi {{
  background: linear-gradient(160deg, rgba(26,38,80,.7), rgba(16,26,58,.85));
  border-color: rgba(138,168,250,.16); box-shadow: none;
}}
.alfa-section--dark .alfa-kpi__label {{ color: var(--on-dark-soft); }}
.alfa-section--dark .alfa-kpi__value {{ color: #fff; }}
.alfa-section--dark .alfa-kpi__note {{ color: var(--on-dark-soft); }}

/* Tabela HTML de dados */
.alfa-table-wrap {{ overflow-x: auto; border-radius: var(--radius); border: 1px solid rgba(138,168,250,.16); }}
.alfa-table {{ width: 100%; border-collapse: collapse; font-size: .9rem; font-variant-numeric: tabular-nums; }}
.alfa-table th {{
  text-align: left; font-size: .68rem; text-transform: uppercase; letter-spacing: .1em;
  color: var(--on-dark-soft); font-weight: 700; padding: 12px 14px;
  background: rgba(10,17,40,.6); position: sticky; top: 0;
}}
.alfa-table td {{ padding: 11px 14px; border-top: 1px solid rgba(138,168,250,.1); color: var(--on-dark); }}
.alfa-table tbody tr {{ transition: background .18s var(--ease); }}
.alfa-table tbody tr:hover {{ background: rgba(138,168,250,.07); }}
.alfa-table .num {{ text-align: right; }}
.alfa-table .pos {{ color: #5fd39b; font-weight: 600; }}
.alfa-table .neg {{ color: #ff8f97; font-weight: 600; }}
.alfa-table th:has(+ th:last-child), .alfa-table td:has(.bar) {{ width: 140px; }}
.alfa-table .bar {{
  display: block; height: 6px; min-width: 8px; border-radius: 99px;
  background: linear-gradient(90deg, var(--blue-500), var(--blue-300));
}}
@media (max-width: 640px) {{ .alfa-table td:has(.bar), .alfa-table th:last-child {{ display: none; }} }}

/* ====================================================================
   8. RE-SKIN DOS WIDGETS DO STREAMLIT
   ==================================================================== */
/* O Streamlit carimba larguras em px nos wrappers (stMarkdown, element
   containers). Dentro dos nossos contêineres com padding isso estoura a caixa,
   então forçamos "100% do pai" — que é sempre o valor correto aqui. */
[data-testid="stMarkdown"] {{ width: 100% !important; }}
[class*="st-key-alfaband_"] [data-testid="stElementContainer"],
[class*="st-key-alfaband_"] [data-testid="stHorizontalBlock"],
[class*="st-key-alfaband_"] [data-testid="stVerticalBlock"],
.st-key-alfa_hero [data-testid="stElementContainer"],
.st-key-alfa_subnav [data-testid="stElementContainer"],
.st-key-alfa_page [data-testid="stElementContainer"],
.st-key-alfa_page [data-testid="stHorizontalBlock"],
.st-key-alfa_page [data-testid="stVerticalBlock"] {{ width: 100% !important; }}

/* --- área útil das páginas da plataforma --- */
.st-key-alfa_page {{
  max-width: var(--maxw); margin: 0 auto;
  padding: clamp(26px, 3.5vw, 40px) clamp(18px, 4vw, 32px) clamp(64px, 8vw, 96px);
}}
.st-key-alfa_page > * {{ width: 100% !important; box-sizing: border-box; }}

/* --- navegação por pills --- */
/* O DOM do st.pills mudou entre gerações do Streamlit: até a 1.5x havia
   `[data-baseweb="button-group"]` e `data-testid="stBaseButton-pills"`; da 1.6x
   em diante o BaseWeb saiu e o estado ativo virou `aria-checked`. Todas as
   regras abaixo cobrem as duas — por isso os seletores duplicados. */
[data-testid="stButtonGroup"] {{ width: 100%; }}
[data-testid="stButtonGroup"] > div,
[data-testid="stButtonGroup"] [data-baseweb="button-group"] {{
  display: flex; flex-wrap: wrap; gap: 10px;
}}
[data-testid="stButtonGroup"] button [data-testid="stMarkdownContainer"],
[data-testid="stButtonGroup"] button p,
[data-testid="stButtonGroup"] button div {{
  white-space: nowrap; overflow: visible; text-overflow: clip; max-width: none;
}}
[data-testid="stButtonGroup"] button {{
  max-width: none !important; width: auto !important; white-space: nowrap;
  border-radius: var(--radius-pill) !important;
  border: 1px solid transparent !important;
  font-size: .87rem !important; font-weight: 600 !important;
  padding: 8px 18px !important; min-height: 0 !important;
  transition: background .18s var(--ease), color .18s var(--ease), border-color .18s var(--ease) !important;
}}
/* Pílulas da topbar: transparentes até serem selecionadas. */
.st-key-alfa_topnav [data-testid="stButtonGroup"] button {{
  background: transparent !important;
  border-color: transparent !important;
  color: var(--on-dark-soft) !important;
  box-shadow: none !important;
}}
.st-key-alfa_topnav [data-testid="stButtonGroup"] button * {{ color: inherit !important; }}
.st-key-alfa_topnav [data-testid="stButtonGroup"] button:hover {{
  background: rgba(138,168,250,.14) !important; color: var(--on-dark) !important;
}}
.st-key-alfa_topnav [data-testid="stButtonGroup"] button[aria-checked="true"],
.st-key-alfa_topnav [data-testid="stButtonGroup"] button[kind$="Active"],
.st-key-alfa_topnav [data-testid="stBaseButton-pillsActive"] {{
  background: var(--blue-500) !important; color: #fff !important;
  border-color: rgba(138,168,250,.5) !important;
  box-shadow: 0 4px 16px rgba(73,121,246,.35) !important;
}}
/* alinhado à esquerda de propósito: com `flex-end` + overflow o primeiro
   item fica inacessível na rolagem horizontal do mobile. */
.st-key-alfa_topnav [data-testid="stButtonGroup"] > div,
.st-key-alfa_topnav [data-testid="stButtonGroup"] [data-baseweb="button-group"] {{
  flex-wrap: nowrap !important; justify-content: flex-start; gap: 16px;
}}

/* Pílulas da sub-navegação (fundo claro) */
.st-key-alfa_subnav [data-testid="stButtonGroup"] button {{
  background: var(--surface) !important; color: var(--ink-soft) !important;
  border-color: var(--border) !important;
}}
.st-key-alfa_subnav [data-testid="stButtonGroup"] button * {{ color: inherit !important; }}
.st-key-alfa_subnav [data-testid="stButtonGroup"] button:hover {{
  border-color: var(--blue-300) !important; color: var(--blue-600) !important;
}}
.st-key-alfa_subnav [data-testid="stButtonGroup"] button[aria-checked="true"],
.st-key-alfa_subnav [data-testid="stButtonGroup"] button[kind$="Active"],
.st-key-alfa_subnav [data-testid="stBaseButton-pillsActive"] {{
  background: var(--navy-950) !important; color: #fff !important;
  border-color: var(--navy-950) !important;
}}
.st-key-alfa_subnav {{ margin-bottom: .4rem; }}

/* --- títulos das páginas da plataforma --- */
.st-key-alfa_page [data-testid="stHeading"] h1 {{ font-size: clamp(1.8rem, 3.4vw, 2.4rem); }}
.st-key-alfa_page [data-testid="stHeading"] h2 {{ font-size: clamp(1.2rem, 2.2vw, 1.45rem); margin-top: 1.6rem; }}
.st-key-alfa_page [data-testid="stHeading"] h3 {{ font-size: 1.02rem; }}
[data-testid="stCaptionContainer"] p {{ color: var(--ink-soft) !important; font-size: .9rem !important; }}

/* --- métricas --- */
[data-testid="stMetric"] {{
  background: var(--surface); border: 1px solid var(--border);
  border-radius: var(--radius); padding: 16px 18px; box-shadow: var(--shadow-sm);
  transition: transform .25s var(--ease), box-shadow .25s var(--ease);
}}
[data-testid="stMetric"]:hover {{ transform: translateY(-3px); box-shadow: var(--shadow-md); }}
[data-testid="stMetricLabel"] p {{
  color: var(--ink-soft) !important; font-size: .72rem !important; font-weight: 700 !important;
  text-transform: uppercase; letter-spacing: .1em;
}}
[data-testid="stMetricValue"] {{
  color: var(--ink) !important; font-size: 1.5rem !important; font-weight: 800 !important;
  letter-spacing: -.03em; font-variant-numeric: tabular-nums;
}}

/* --- botões --- */
[data-testid="stButton"] > button, [data-testid="stFormSubmitButton"] > button {{
  border-radius: var(--radius-pill); border: 1px solid var(--border);
  background: var(--surface); color: var(--blue-600);
  font-size: .89rem; font-weight: 600; min-height: 2.6rem; padding: 0 20px;
  box-shadow: var(--shadow-sm);
  transition: transform .18s var(--ease), background .18s var(--ease), border-color .18s var(--ease), box-shadow .18s var(--ease);
}}
[data-testid="stButton"] > button:hover, [data-testid="stFormSubmitButton"] > button:hover {{
  background: #f4f7ff; border-color: var(--blue-300); color: var(--blue-600); transform: translateY(-1px);
}}
[data-testid="stButton"] > button[kind="primary"],
[data-testid="stFormSubmitButton"] > button[kind="primaryFormSubmit"] {{
  background: linear-gradient(135deg, var(--blue-500), var(--blue-600)) !important;
  border-color: transparent !important; color: #fff !important;
  box-shadow: 0 6px 20px rgba(73,121,246,.28) !important;
}}
[data-testid="stButton"] > button[kind="primary"]:hover,
[data-testid="stFormSubmitButton"] > button[kind="primaryFormSubmit"]:hover {{
  transform: translateY(-2px); box-shadow: 0 12px 30px rgba(73,121,246,.36) !important;
}}
[data-testid="stButton"] > button p, [data-testid="stFormSubmitButton"] > button p {{ font-weight: 600; }}

/* --- inputs --- */
[data-baseweb="base-input"], [data-baseweb="select"] > div,
[data-testid="stTextInputRootElement"], [data-testid="stTextAreaRootElement"],
[data-testid="stSelectbox"] > div:last-child,
[data-testid="stNumberInputContainer"] {{
  border-radius: 12px !important; border: 1px solid var(--border) !important;
  background: var(--surface) !important; box-shadow: none !important;
}}
[data-baseweb="base-input"]:focus-within,
[data-testid="stTextInputRootElement"]:focus-within,
[data-testid="stTextAreaRootElement"]:focus-within,
[data-testid="stNumberInputContainer"]:focus-within {{
  border-color: var(--blue-300) !important; box-shadow: 0 0 0 3px rgba(73,121,246,.12) !important;
}}
input, select, textarea, [data-baseweb="select"], [data-testid="stSelectbox"] {{
  font-family: var(--font) !important;
}}
[data-testid="stWidgetLabel"] p {{
  font-size: .78rem !important; font-weight: 600 !important; color: var(--ink-soft) !important;
  letter-spacing: .02em;
}}

/* --- dataframes / tabelas --- */
[data-testid="stDataFrame"], [data-testid="stTable"] {{
  border: 1px solid var(--border) !important; border-radius: var(--radius) !important; overflow: hidden;
  box-shadow: var(--shadow-sm);
}}

/* --- expander --- */
[data-testid="stExpander"] {{
  border: 1px solid var(--border) !important; border-radius: var(--radius) !important;
  background: var(--surface) !important; box-shadow: var(--shadow-sm) !important; overflow: hidden;
}}
[data-testid="stExpander"] summary {{ font-weight: 600; font-size: .92rem; }}

/* --- abas --- */
[data-testid="stTabs"] [data-baseweb="tab-list"],
[data-testid="stTabs"] [role="tablist"] {{ gap: 4px; border-bottom: 1px solid var(--border); }}
[data-testid="stTabs"] button[data-baseweb="tab"],
[data-testid="stTabs"] [data-testid="stTab"] {{
  font-size: .89rem !important; font-weight: 600 !important; color: var(--ink-soft);
  padding: 10px 16px;
}}
[data-testid="stTabs"] button[data-baseweb="tab"][aria-selected="true"],
[data-testid="stTabs"] [data-testid="stTab"][aria-selected="true"] {{
  color: var(--blue-600); border-bottom-color: var(--blue-500);
}}
[data-testid="stTabs"] [data-baseweb="tab-highlight"] {{ background: var(--blue-500); height: 2.5px; }}

/* --- slider --- */
[data-testid="stSlider"] [role="slider"] {{ box-shadow: 0 2px 8px rgba(73,121,246,.35); }}

/* --- gráficos --- */
[data-testid="stPlotlyChart"], [data-testid="stElementContainer"]:has(> [data-testid="stPlotlyChart"]) {{
  border-radius: var(--radius);
}}
[data-testid="stPlotlyChart"] {{
  background: var(--surface); border: 1px solid var(--border);
  border-radius: var(--radius); padding: 10px 6px 4px; box-shadow: var(--shadow-sm);
  overflow: hidden;
}}
/* o Streamlit escreve `background` inline no <svg> mesmo com theme=None */
[data-testid="stPlotlyChart"] .main-svg {{ background: transparent !important; }}

/* --- alertas --- */
[data-testid="stAlert"] {{ border-radius: var(--radius); border: 1px solid var(--border); }}

/* Título de seção reaproveitado das telas antigas */
.alfa-section-title {{
  color: var(--ink-soft); font-size: .7rem; font-weight: 800;
  letter-spacing: .12em; text-transform: uppercase; margin: 1.4rem 0 .7rem;
}}
/* Compatibilidade com as classes usadas nas páginas herdadas */
.alfa-kpi-card {{
  background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius);
  padding: 16px 18px; box-shadow: var(--shadow-sm); height: 100%;
}}
.alfa-kpi-label {{
  color: var(--ink-soft); font-size: .7rem; font-weight: 700;
  letter-spacing: .1em; text-transform: uppercase; margin-bottom: .5rem;
}}
.alfa-kpi-value {{
  color: var(--ink); font-size: 1.45rem; font-weight: 800; letter-spacing: -.03em;
  line-height: 1.15; margin-bottom: .25rem; font-variant-numeric: tabular-nums;
}}
.alfa-kpi-note {{ color: var(--blue-600); font-size: .82rem; font-weight: 500; }}
.alfa-kpi-trend.positive {{ color: var(--pos); background: rgba(46,158,107,.12); }}
.alfa-kpi-trend.negative {{ color: var(--neg); background: rgba(214,69,80,.12); }}


/* --- faixas full-bleed que contêm widgets do Streamlit ------------------ */
[class*="st-key-alfaband_"] {{
  width: 100%;
  padding: clamp(46px, 7vw, 92px) clamp(20px, 4vw, 32px);
  position: relative; overflow: hidden;
  align-items: center;
}}
[class*="st-key-alfaband_"] > * {{ width: 100% !important; max-width: var(--maxw); box-sizing: border-box; }}
[class*="st-key-alfaband_dark"] {{ background: var(--navy-950); color: var(--on-dark); }}
[class*="st-key-alfaband_dark"] h1, [class*="st-key-alfaband_dark"] h2,
[class*="st-key-alfaband_dark"] h3 {{ color: var(--on-dark) !important; }}
[class*="st-key-alfaband_dark"] p, [class*="st-key-alfaband_dark"] .alfa-lead {{ color: var(--on-dark-soft); }}
[class*="st-key-alfaband_dark"] .alfa-eyebrow {{ color: var(--blue-300); }}
[class*="st-key-alfaband_dark"] [data-testid="stCaptionContainer"] p {{ color: var(--on-dark-soft) !important; }}
[class*="st-key-alfaband_light"] {{ background: var(--bg); color: var(--ink); }}
[class*="st-key-alfaband_surface"] {{ background: var(--surface); color: var(--ink); }}

/* Hero: faixa escura com respiro e alinhamento próprios */
.st-key-alfa_hero {{
  width: 100%; position: relative; overflow: hidden;
  padding: calc(var(--nav-h) + clamp(52px, 8vw, 96px)) clamp(20px, 4vw, 32px) clamp(56px, 8vw, 96px);
  margin-top: calc(var(--nav-h) * -1);
  background: var(--navy-950); color: var(--on-dark);
  align-items: center; text-align: center; gap: 0;
  min-height: min(760px, 92vh); justify-content: center;
}}
.st-key-alfa_hero > * {{ width: 100% !important; max-width: 820px; box-sizing: border-box; z-index: 1; }}
.st-key-alfa_hero [data-testid="stButton"] > button {{
  width: 100%; border-radius: var(--radius-pill); font-weight: 600; min-height: 2.9rem;
}}
.st-key-alfa_hero [data-testid="stButton"] > button[kind="secondary"] {{
  background: rgba(138,168,250,.10); border-color: rgba(138,168,250,.45); color: var(--blue-300);
  box-shadow: none;
}}
.st-key-alfa_hero [data-testid="stButton"] > button[kind="secondary"]:hover {{
  background: rgba(138,168,250,.2); border-color: var(--blue-300); color: #fff;
}}

/* Botões de CTA dentro de faixas escuras (fora do hero) */
[class*="st-key-alfaband_dark"] [data-testid="stButton"] > button[kind="secondary"] {{
  background: rgba(138,168,250,.10); border-color: rgba(138,168,250,.45); color: var(--blue-300); box-shadow: none;
}}
[class*="st-key-alfaband_dark"] [data-testid="stButton"] > button[kind="secondary"]:hover {{
  background: rgba(138,168,250,.2); border-color: var(--blue-300); color: #fff;
}}
[class*="st-key-alfaband_dark"] [data-testid="stMetric"] {{
  background: linear-gradient(160deg, rgba(26,38,80,.72), rgba(16,26,58,.86));
  border-color: rgba(138,168,250,.16); box-shadow: none;
}}
[class*="st-key-alfaband_dark"] [data-testid="stMetricValue"] {{ color: #fff !important; }}
[class*="st-key-alfaband_dark"] [data-testid="stMetricLabel"] p {{ color: var(--on-dark-soft) !important; }}
[class*="st-key-alfaband_dark"] [data-testid="stPlotlyChart"] {{
  background: rgba(16,26,58,.55); border-color: rgba(138,168,250,.16); box-shadow: none;
}}
[class*="st-key-alfaband_dark"] [data-testid="stExpander"] {{
  background: rgba(16,26,58,.6) !important; border-color: rgba(138,168,250,.16) !important; box-shadow: none !important;
}}
[class*="st-key-alfaband_dark"] [data-testid="stExpander"] summary, 
[class*="st-key-alfaband_dark"] [data-testid="stExpander"] p {{ color: var(--on-dark) !important; }}

/* Uma faixa que precisa "encostar" na anterior */
.alfa-flush {{ padding-top: 0 !important; }}


/* --- cartão de destaque das telas Comparador / Projeções --------------- */
.alfa-tickercard {{
  background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius);
  padding: 16px 18px; box-shadow: var(--shadow-sm); height: 100%;
  transition: transform .25s var(--ease), box-shadow .25s var(--ease);
}}
.alfa-tickercard:hover {{ transform: translateY(-3px); box-shadow: var(--shadow-md); }}
.alfa-tickercard__label {{
  color: var(--ink-soft); font-size: .7rem; font-weight: 700;
  letter-spacing: .1em; text-transform: uppercase; margin-bottom: .5rem;
}}
.alfa-tickercard__value {{
  color: var(--ink); font-size: 1.6rem; font-weight: 800; line-height: 1.1;
  letter-spacing: -.03em; margin-bottom: .6rem;
}}
.alfa-tickercard__pill {{
  display: inline-flex; align-items: center; border-radius: var(--radius-pill);
  padding: .28rem .7rem; font-size: .85rem; font-weight: 700;
}}
.alfa-tickercard__pill.positive {{ background: rgba(46,158,107,.13); color: var(--pos); }}
.alfa-tickercard__pill.negative {{ background: rgba(214,69,80,.12); color: var(--neg); }}

/* --- CTA em contorno sobre fundo claro --------------------------------- */
.alfa-btn--outline {{ border-color: var(--blue-500); color: var(--blue-500) !important; }}
.alfa-btn--outline:hover {{ background: var(--blue-500); color: #fff !important; transform: translateY(-2px); }}

/* --- sub-navegação da plataforma (fica colada abaixo da topbar) --------- */
.st-key-alfa_subnav {{
  position: sticky; top: var(--nav-h); z-index: 900;
  background: rgba(242,241,236,.94);
  backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px);
  border-bottom: 1px solid var(--border);
  width: 100%; padding: 12px clamp(20px, 4vw, 32px);
  align-items: center;
}}
.st-key-alfa_subnav > * {{ width: 100% !important; max-width: var(--maxw); box-sizing: border-box; overflow-x: auto; scrollbar-width: none; }}
.st-key-alfa_subnav > *::-webkit-scrollbar {{ display: none; }}
.st-key-alfa_subnav [data-testid="stButtonGroup"] > div,
.st-key-alfa_subnav [data-testid="stButtonGroup"] [data-baseweb="button-group"] {{
  flex-wrap: nowrap !important; gap: 12px;
}}
.st-key-alfa_subnav [data-testid="stButtonGroup"] button {{ white-space: nowrap; }}

/* ====================================================================
   9. RODAPÉ
   ==================================================================== */
.alfa-footer {{
  width: 100%;
  background: var(--navy-950); color: var(--on-dark-soft);
  padding: clamp(44px, 6vw, 68px) 0 28px; font-size: .92rem;
}}
.alfa-footer__grid {{ display: grid; grid-template-columns: 2fr 1fr 1fr; gap: clamp(24px, 4vw, 48px); }}
.alfa-footer h4 {{
  color: var(--on-dark); font-size: .72rem; letter-spacing: .14em;
  text-transform: uppercase; margin-bottom: .9rem;
}}
.alfa-footer a {{ color: var(--on-dark-soft); }}
.alfa-footer a:hover {{ color: var(--on-dark); }}
.alfa-footer ul {{ list-style: none !important; margin: 0 !important; padding: 0 !important; }}
.alfa-footer li {{ margin: 0 0 .5rem 0 !important; padding: 0 !important; list-style: none; }}
.alfa-footer__bottom {{
  border-top: 1px solid rgba(138,168,250,.14); margin-top: clamp(28px, 4vw, 44px);
  padding-top: 20px; display: flex; justify-content: space-between; gap: 12px;
  flex-wrap: wrap; font-size: .82rem;
}}
.alfa-social {{ display: flex; gap: 14px; }}
.alfa-social a {{
  width: 36px; height: 36px; border-radius: 50%;
  border: 1px solid rgba(138,168,250,.24);
  display: flex; align-items: center; justify-content: center;
  transition: background .2s var(--ease), transform .2s var(--ease), border-color .2s var(--ease);
}}
.alfa-social a:hover {{ background: var(--blue-500); border-color: var(--blue-500); transform: translateY(-2px); }}
.alfa-social svg {{ width: 17px; height: 17px; fill: currentColor; }}

/* ====================================================================
   10. RESPONSIVO
   ==================================================================== */
@media (max-width: 900px) {{
  .alfa-split {{ grid-template-columns: 1fr; }}
  .alfa-footer__grid {{ grid-template-columns: 1fr 1fr; }}
}}
@media (max-width: 640px) {{
  :root {{ --nav-h: {NAV_H_MOBILE}px; }}
  .st-key-alfa_topnav .alfa-brand-txt,
  .st-key-alfa_topnav .alfa-brand-sub {{ display: none; }}
  .alfa-footer__grid {{ grid-template-columns: 1fr; }}
  .alfa-tl-step {{ padding-left: 54px !important; }}
  .alfa-timeline::before {{ left: 17px; }}
  .alfa-tl-step::before {{ width: 36px; height: 36px; font-size: .85rem; }}
  .st-key-alfa_hero {{ min-height: 0; }}
  .st-key-alfa_topnav [data-testid="stBaseButton-pills"],
  .st-key-alfa_topnav [data-testid="stBaseButton-pillsActive"] {{ font-size: .8rem !important; padding: 6px 13px !important; }}
  .alfa-ctas {{ flex-direction: column; align-items: stretch; }}
  .alfa-btn {{ justify-content: center; }}
  /* Colunas do Streamlit empilham no mobile — reduz o respiro entre elas */
  [data-testid="stHorizontalBlock"] {{ gap: .6rem; }}
}}
/* ====================================================================
   11. OVERRIDES FINAIS
   Ficam por último de propósito: sobrescrevem as regras genéricas de
   largura da seção 8, que são mais amplas mas têm a mesma especificidade.
   ==================================================================== */

/* Componentes HTML dentro das faixas escuras montadas com st.container(key=…):
   repetem o que `.alfa-section--dark` já faz para as faixas 100% HTML. */
[class*="st-key-alfaband_dark"] .alfa-card,
[class*="st-key-alfaband_dark"] .alfa-kpi,
[class*="st-key-alfaband_dark"] .alfa-tickercard {{
  background: linear-gradient(160deg, rgba(26,38,80,.72), rgba(16,26,58,.86));
  border-color: rgba(138,168,250,.16); box-shadow: none;
}}
[class*="st-key-alfaband_dark"] .alfa-card:hover,
[class*="st-key-alfaband_dark"] .alfa-kpi:hover {{
  border-color: rgba(138,168,250,.42); box-shadow: 0 18px 44px rgba(0,0,0,.34);
}}
[class*="st-key-alfaband_dark"] .alfa-card h3 {{ color: var(--blue-300); }}
[class*="st-key-alfaband_dark"] .alfa-card p,
[class*="st-key-alfaband_dark"] .alfa-card__tag,
[class*="st-key-alfaband_dark"] .alfa-kpi__label,
[class*="st-key-alfaband_dark"] .alfa-kpi__note,
[class*="st-key-alfaband_dark"] .alfa-tickercard__label {{ color: var(--on-dark-soft); }}
[class*="st-key-alfaband_dark"] .alfa-kpi__value,
[class*="st-key-alfaband_dark"] .alfa-tickercard__value {{ color: #fff; }}
[class*="st-key-alfaband_dark"] .alfa-kpi__note.up {{ color: #5fd39b; }}
[class*="st-key-alfaband_dark"] .alfa-kpi__note.down {{ color: #ff8f97; }}
[class*="st-key-alfaband_dark"] .alfa-stat__value {{
  background: linear-gradient(135deg, #ffffff 12%, var(--blue-300) 96%);
  -webkit-background-clip: text; background-clip: text; color: transparent;
}}
[class*="st-key-alfaband_dark"] .alfa-stat__label {{ color: var(--on-dark-soft); }}
[class*="st-key-alfaband_dark"] .alfa-tag {{ border-color: var(--blue-300); color: var(--blue-300); }}
[class*="st-key-alfaband_dark"] .alfa-tag:hover {{ background: var(--blue-300); color: var(--navy-950); }}
[class*="st-key-alfaband_dark"] .alfa-chip--info {{ background: rgba(138,168,250,.16); color: var(--blue-300); }}
[class*="st-key-alfaband_dark"] .alfa-tl-step::before {{ background: var(--navy-900); color: var(--blue-300); }}
[class*="st-key-alfaband_dark"] .alfa-tl-step p {{ color: var(--on-dark-soft); }}

/* Componentes HTML dentro das faixas escuras montadas com st.container(key=…):
   repetem o que `.alfa-section--dark` já faz para as faixas 100% HTML. */
[class*="st-key-alfaband_dark"] .alfa-card,
[class*="st-key-alfaband_dark"] .alfa-kpi,
[class*="st-key-alfaband_dark"] .alfa-tickercard {{
  background: linear-gradient(160deg, rgba(26,38,80,.72), rgba(16,26,58,.86));
  border-color: rgba(138,168,250,.16); box-shadow: none;
}}
[class*="st-key-alfaband_dark"] .alfa-card:hover,
[class*="st-key-alfaband_dark"] .alfa-kpi:hover {{
  border-color: rgba(138,168,250,.42); box-shadow: 0 18px 44px rgba(0,0,0,.34);
}}
[class*="st-key-alfaband_dark"] .alfa-card h3 {{ color: var(--blue-300); }}
[class*="st-key-alfaband_dark"] .alfa-card p,
[class*="st-key-alfaband_dark"] .alfa-card__tag,
[class*="st-key-alfaband_dark"] .alfa-kpi__label,
[class*="st-key-alfaband_dark"] .alfa-kpi__note,
[class*="st-key-alfaband_dark"] .alfa-tickercard__label {{ color: var(--on-dark-soft); }}
[class*="st-key-alfaband_dark"] .alfa-kpi__value,
[class*="st-key-alfaband_dark"] .alfa-tickercard__value {{ color: #fff; }}
[class*="st-key-alfaband_dark"] .alfa-kpi__note.up {{ color: #5fd39b; }}
[class*="st-key-alfaband_dark"] .alfa-kpi__note.down {{ color: #ff8f97; }}
[class*="st-key-alfaband_dark"] .alfa-stat__value {{
  background: linear-gradient(135deg, #ffffff 12%, var(--blue-300) 96%);
  -webkit-background-clip: text; background-clip: text; color: transparent;
}}
[class*="st-key-alfaband_dark"] .alfa-stat__label {{ color: var(--on-dark-soft); }}
[class*="st-key-alfaband_dark"] .alfa-tag {{ border-color: var(--blue-300); color: var(--blue-300); }}
[class*="st-key-alfaband_dark"] .alfa-tag:hover {{ background: var(--blue-300); color: var(--navy-950); }}
[class*="st-key-alfaband_dark"] .alfa-chip--info {{ background: rgba(138,168,250,.16); color: var(--blue-300); }}
[class*="st-key-alfaband_dark"] .alfa-tl-step::before {{ background: var(--navy-900); color: var(--blue-300); }}
[class*="st-key-alfaband_dark"] .alfa-tl-step p {{ color: var(--on-dark-soft); }}

/* O rótulo dos botões vive num <p>: a regra de parágrafo das faixas escuras
   não pode sobrescrever a cor definida pelo próprio botão. */
[class*="st-key-alfaband_dark"] [data-testid="stButton"] > button p,
[class*="st-key-alfaband_dark"] [data-testid="stFormSubmitButton"] > button p,
[class*="st-key-alfaband_dark"] [data-testid="stBaseButton-pills"] p,
[class*="st-key-alfaband_dark"] [data-testid="stBaseButton-pillsActive"] p {{ color: inherit !important; }}

/* O Streamlit envolve cada container num wrapper que continua no fluxo mesmo
   com a topbar `position: fixed` — e o `gap` do bloco vertical abria uma faixa
   clara logo abaixo da barra. Tiramos o wrapper do fluxo. */
[data-testid="stVerticalBlockBorderWrapper"]:has(> div > .st-key-alfa_topnav),
[data-testid="stLayoutWrapper"]:has(> .st-key-alfa_topnav),
[data-testid="stLayoutWrapper"]:has(> div > .st-key-alfa_topnav) {{
  position: absolute; height: 0; overflow: visible;
}}

/* number_input: a borda tem que envolver campo + setas, não só o campo */
[data-testid="stNumberInputContainer"] {{ overflow: hidden; }}
[data-testid="stNumberInputContainer"] [data-baseweb="input"],
[data-testid="stNumberInputContainer"] [data-baseweb="base-input"],
[data-testid="stNumberInputContainer"] > div:first-child {{
  border: 0 !important; border-radius: 0 !important; background: transparent !important;
  box-shadow: none !important;
}}
[data-testid="stNumberInputStepUp"], [data-testid="stNumberInputStepDown"] {{
  color: var(--ink-soft); background: transparent;
}}
[data-testid="stNumberInputStepUp"]:hover, [data-testid="stNumberInputStepDown"]:hover {{
  color: var(--blue-600); background: rgba(73,121,246,.08);
}}

/* As faixas do site têm que se encostar: o `gap` do bloco vertical raiz
   abria uma listra da cor de fundo entre uma seção e a seguinte. */
.st-key-alfa_root {{ gap: 0 !important; }}
.st-key-alfa_root [data-testid="stVerticalBlock"] {{ gap: .85rem; }}

/* Nas barras de navegação o grupo é nowrap: as pílulas não podem encolher,
   senão o rótulo é cortado. A barra inteira rola na horizontal. */
.st-key-alfa_topnav [data-testid="stButtonGroup"] button,
.st-key-alfa_subnav [data-testid="stButtonGroup"] button {{
  flex: 0 0 auto !important;
}}

/* Linha de CTA: botões reais do Streamlit lado a lado e centralizados */
[class*="st-key-alfa_cta_"] {{
  flex-direction: row !important; justify-content: center; align-items: center;
  gap: 14px; flex-wrap: wrap; margin-top: 10px;
}}
[class*="st-key-alfa_cta_"] > [data-testid="stElementContainer"] {{
  width: auto !important; flex: 0 0 auto;
}}
[class*="st-key-alfa_cta_"] [data-testid="stButton"],
[class*="st-key-alfa_cta_"] [data-testid="stButton"] > button {{
  width: auto !important; white-space: nowrap;
}}
[class*="st-key-alfa_cta_"] [data-testid="stButton"] > button {{
  min-height: 2.9rem; padding: 0 30px;
}}
@media (max-width: 640px) {{
  [class*="st-key-alfa_cta_"] {{ flex-direction: column !important; align-items: stretch; }}
  [class*="st-key-alfa_cta_"] > [data-testid="stElementContainer"] {{ width: 100% !important; }}
  [class*="st-key-alfa_cta_"] [data-testid="stButton"],
  [class*="st-key-alfa_cta_"] [data-testid="stButton"] > button {{ width: 100% !important; }}
}}
"""


def inject_global_css() -> None:
    """Injeta a folha global uma única vez por rerun."""
    st.markdown(f"<style>{_css()}</style>", unsafe_allow_html=True)
