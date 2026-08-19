# ALFA — site institucional + plataforma quant (Streamlit)

Um único app Streamlit que junta:

- **o site institucional** (Início, O Fundo, Membros, Atividades, Processo Seletivo),
  com o conteúdo que hoje vive no repositório `alfapucrio.github.io`;
- **a plataforma quant** (Carteira, Histórico, Risco, Stress Test, Markowitz,
  Black-Litterman, APT, Análise de Ativos, Comparador, Projeções Quant), com os
  mesmos modelos e cálculos do app Streamlit atual.

O visual é o da identidade ALFA (navy `#0a1128`, azul `#4979f6`, off-white
`#f2f1ec`, tipografia Inter), num layout de landing page: faixas full-bleed
alternando claro/escuro, barra fixa de navegação, animações de entrada e
gráficos com tema próprio.

---

## Rodar localmente

```bash
pip install -r requirements.txt
```

```bash
streamlit run app.py
```

## Publicar no Streamlit Community Cloud

1. Suba esta pasta para um repositório no GitHub (pode ser a raiz do repo).
2. Em [share.streamlit.io](https://share.streamlit.io) → **New app**.
3. Aponte para o repositório e use `app.py` como *Main file path*.
4. **Deploy**. O `requirements.txt` e o `.streamlit/config.toml` já estão prontos.

Links diretos funcionam via query string: `…/?p=fundo`, `…/?p=plataforma`,
`…/?p=processo` etc.

---

## Editar o conteúdo (sem programar)

Todo o texto variável do site está em [`content/`](content/), no mesmo formato
do site antigo. Basta editar o JSON e fazer commit.

| Arquivo | O que controla |
|---|---|
| `content/membros.json` | Membros por diretoria (nome, cargo, foto, LinkedIn) |
| `content/eventos.json` | Calendário do semestre, na página Processo Seletivo |
| `content/atividades.json` | Visitas institucionais e competições |
| `content/highlights.json` | Números da página inicial (+30 membros, +40 projetos…) |

**Fotos de membros:** suba o arquivo em `assets/membros/` (quadrado, ~400×400px)
e escreva o nome do arquivo no campo `"foto"` do membro.
**Fotos de atividades:** o mesmo, em `assets/atividades/` (paisagem, ~1200×750px).

Textos fixos (filosofia do fundo, jornada do membro, descrição das diretorias)
ficam no topo de cada arquivo em `site_pages/` — são listas Python simples,
fáceis de editar.

---

## Dados de mercado

A carteira e o histórico vêm dos mesmos CSVs do app atual, na raiz do projeto:

- `portfolio_data.csv` — posições, pesos e preços de fechamento;
- `historico_data.csv` — série histórica consolidada (ativos + IBOV + CDI + títulos).

Para recalcular o histórico: **Plataforma → Carteira → Atualizar Histórico**
(baixa preços do Yahoo Finance, CDI do Banco Central e títulos públicos). A
página **O Fundo** também oferece esse botão quando percebe que o histórico não
cobre as posições atuais.

> ⚠️ No Streamlit Community Cloud o disco é efêmero: alterações feitas pela
> interface (carteira, histórico) valem para a sessão e se perdem no próximo
> deploy. Para persistir, faça commit dos CSVs no repositório.

> ⚠️ A tela **Carteira** permite editar posições e o app é público. Se quiser
> deixá-la só para a diretoria, o caminho mais simples é remover a entrada
> `carteira` da tupla em `site_pages/plataforma.py` e manter a edição num app
> separado/privado.

---

## Estrutura

```
app.py              casca do site: config, CSS, topbar, roteador, rodapé
theme/
  tokens.py         ÚNICA fonte de verdade das cores da identidade
  styles.py         folha de estilo global (a aparência inteira vive aqui)
  components.py     blocos em HTML (hero, cards, timeline, KPIs, tabelas…)
  plotly_theme.py   templates "alfa" e "alfa_dark" para todos os gráficos
site_pages/         páginas institucionais + hub da plataforma
ui/                 telas dos modelos (herdadas do app atual)
services/ core/     cálculos financeiros e acesso a dados
content/            conteúdo editável (JSON)
assets/             logos, ondas e fotos
tests/smoke_test.py renderiza todas as páginas e falha se alguma quebrar
```

### Mexer no design

- **Cores:** só em `theme/tokens.py`. O CSS e os gráficos leem de lá.
- **Espaçamento, sombras, raios, breakpoints:** bloco `1. TOKENS` de
  `theme/styles.py`.
- **Tipografia de display:** a variável `--font-display` (também no bloco de
  tokens) permite trocar os títulos por uma serifada sem tocar no resto.
- **Gráficos:** `theme/plotly_theme.py`. Todos os `st.plotly_chart` usam
  `theme=None` justamente para o template ALFA prevalecer sobre o do Streamlit.

---

## Testes

```bash
python tests/smoke_test.py
```

Renderiza as 5 páginas institucionais e as 10 ferramentas num runtime headless
e falha se alguma levantar exceção. Para ver o traceback completo (o app roda
com `showErrorDetails=false`), rode com
`STREAMLIT_CLIENT_SHOW_ERROR_DETAILS=true`.

---

Conteúdo educacional. Nada neste site constitui recomendação de investimento.
