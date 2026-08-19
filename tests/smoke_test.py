"""Smoke test do site ALFA.

Renderiza cada página institucional e cada ferramenta da plataforma num
runtime headless do Streamlit e falha se alguma levantar exceção.

    cd ALFA-site && python tests/smoke_test.py

Dica: para ver o traceback completo (o app roda com `showErrorDetails=false`),
rode com `STREAMLIT_CLIENT_SHOW_ERROR_DETAILS=true`.
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)  # AppTest não adiciona o diretório do app ao sys.path
from streamlit.testing.v1 import AppTest

PAGES = ["inicio", "fundo", "newsletter", "membros", "alumni", "atividades", "processo"]
TOOLS = ["carteira", "historico", "risco", "stress", "markowitz", "bl", "apt",
         "ativos", "comparador", "quant"]

fails = []

def run(label, state):
    at = AppTest.from_file("app.py", default_timeout=180)
    for k, v in state.items():
        at.session_state[k] = v
    try:
        at.run()
    except Exception as exc:
        fails.append((label, f"{type(exc).__name__}: {exc}"))
        print(f"  ERRO  {label}: {type(exc).__name__}: {exc}")
        return
    if at.exception:
        msgs = "; ".join(e.value for e in at.exception)
        fails.append((label, msgs))
        print(f"  ERRO  {label}: {msgs}")
    else:
        n_err = len(at.error)
        print(f"  ok    {label}" + (f"  ({n_err} st.error na tela)" if n_err else ""))

print("== páginas institucionais ==")
for page in PAGES:
    run(page, {"page": page})

print("== ferramentas da plataforma ==")
for tool in TOOLS:
    run(f"plataforma/{tool}", {"page": "plataforma", "tool": tool})

print()
print(f"RESULTADO: {len(fails)} falha(s)")
sys.exit(1 if fails else 0)
