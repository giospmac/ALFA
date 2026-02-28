# ALFA

Aplicação de análise de investimentos com duas interfaces:

- `ALFA.py`: interface desktop legada em PyQt6.
- `app.py`: interface web em Streamlit, pronta para publicação no Streamlit Community Cloud.

## Estrutura

```text
.
├── ALFA.py
├── app.py
├── core/
│   └── portfolio_repository.py
├── services/
│   └── market_data.py
├── ui/
│   ├── asset_analysis.py
│   └── home.py
└── requirements.txt
```

## O que foi reaproveitado

O desktop legado em `ALFA.py` continua intacto. A versão Streamlit foi expandida para cobrir o mesmo fluxo principal do app PyQt:

- montagem do portfólio com acoes e titulos publicos
- persistencia em `portfolio_data.csv` e `historico_data.csv`
- historico consolidado com `IBOVESPA` e `CDI`
- pagina de graficos com comparacao vs benchmarks, contribuicao, drawdown, Monte Carlo e volatilidade
- pagina de indicadores de risco com rentabilidade acumulada, VaR, CVaR, CAPM, alfa, beta, correlacao e indices de desempenho
- pagina de fronteira eficiente de Markowitz
- pagina de analise individual de ativos via Yahoo Finance

## Como criar o ambiente virtual

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Windows (PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

## Como instalar dependências

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Como rodar localmente

### App web em Streamlit

```bash
streamlit run app.py
```

Depois abra a URL exibida no terminal, normalmente `http://localhost:8501`.

### App desktop legado

```bash
python ALFA.py
```

## Funcionalidades da interface Streamlit

### Portfólio

- adiciona acoes e titulos publicos diretamente pela interface
- recalcula quantidades com base no PL informado
- atualiza historico consolidado e benchmarks
- exibe tabela da carteira, caixa livre e evolucao normalizada

### Graficos

- retorno acumulado do portfolio contra `IBOVESPA` em 1 mes, 1 ano e 5 anos
- retorno acumulado do portfolio contra `CDI` em 1 mes, 1 ano e 5 anos
- contribuicao de retorno por ativo
- drawdown historico
- simulacao de Monte Carlo com VaR e CVaR
- volatilidade anualizada rolling de 21 dias uteis

### Indicadores de Risco

- rentabilidade mensal e anual acumulada
- media, variancia e volatilidade por ativo e portfolio
- VaR e CVaR em 95% e 99%
- CAPM, alfa, beta e correlacao com o mercado
- Sharpe, Sortino, Treynor e Tracking Error
- matriz de correlacao entre os ativos

### Fronteira Eficiente

- simulacao de carteiras
- destaque para maximo Sharpe
- destaque para minima volatilidade
- tabela de pesos otimizados

### Análise de Ativos

- consulta ticker via Yahoo Finance com `yfinance`
- mostra preco atual, beta, variacao do dia, P/L, EV/EBITDA, market cap, margem EBITDA e crescimento de receita

## Deploy no Streamlit Community Cloud

1. Suba o projeto para um repositório no GitHub.
2. Acesse `https://share.streamlit.io/`.
3. Clique em `Create app`.
4. Selecione o repositório, branch e o arquivo principal `app.py`.
5. Confirme que o `requirements.txt` está na raiz do projeto.
6. Clique em `Deploy`.

Se fizer novas alterações depois, basta enviar para o GitHub e usar `Reboot app` ou `Redeploy` no painel do Streamlit.

## Secrets e credenciais

Hoje o app não exige segredos para funcionar. Se no futuro você adicionar APIs privadas:

1. Crie o arquivo `.streamlit/secrets.toml` localmente.
2. Não commit esse arquivo.
3. No Streamlit Community Cloud, configure os mesmos valores em `App settings > Secrets`.

Exemplo:

```toml
[api]
token = "seu_token_aqui"
```

## Observações

- O Streamlit Cloud instalará as dependências listadas em `requirements.txt`.
- O fluxo principal do app web não depende de CSV.
- O histórico e os preços do portfólio são consultados diretamente no Yahoo Finance.
