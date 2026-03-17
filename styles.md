# Identidade Visual ALFA (Style Guide)

Este documento define as diretrizes de design e identidade visual simples, moderna e bonita para a aplicação ALFA. Todos os novos componentes, gráficos e páginas devem seguir estas definições para garantir uma experiência de usuário (UX) coesa e elegante.

## 1. Tipografia

- **Fonte Principal:** `Inter` (sans-serif)
- A fonte deve estar configurada globalmente na aplicação via injeção de CSS em `app.py`.
- **Pesos Utilizados:**
  - `400` (Regular) - Para textos gerais e tabelas.
  - `500` (Medium) - Para botões, abas e labels secundárias.
  - `600` (SemiBold) - Para subtítulos e valores de KPIs.
  - `700` (Bold) - Para títulos principais e destaque de valores métricos.

## 2. Paleta de Cores (Tokens Globais)

As seguintes variáveis CSS devem reger a interface. Ao criar componentes com HTML/CSS, utilize as variáveis abaixo. Ao usar bibliotecas como Matplotlib (gráficos) que exigem Hex, utilize os valores exatos:

- **Bases (Fundos e Superfícies):**
  - Fundo do App (`--alfa-bg`): `#F8F9FA`
  - Superfícies/Cards (`--alfa-surface`): `#FFFFFF`
  - Bordas e Divisores (`--alfa-border`): `#E5E7EB`

- **Textos:**
  - Texto Principal (`--alfa-text`): `#111827` (Quase preto, alta legibilidade)
  - Texto Mutado/Secundário (`--alfa-muted`): `#6B7280` (Legendas, labels de eixos de gráficos)
  - Texto Suave (`--alfa-soft`): `#9CA3AF` (Subtítulos de navegação, placeholders)

- **Cores de Destaque e Ação:**
  - Ação Principal/Accent (`--alfa-accent`): `#4979f6` (Azul ALFA moderno)
  - Ação Secundária/Hover (`--alfa-accent-2`): `#2f5adf`
  - Fundo sutil de destaque (Azul bem claro): `#EFF6FF`

- **Cores Semânticas:**
  - Positivo (`--alfa-positive`): `#059669` (Verde esmeralda - para retornos e lucros)
  - Fundo Positivo: `#ECFDF5`
  - Negativo (`--alfa-negative`): `#EF4444` (Vermelho - para perdas e riscos altos)
  - Fundo Negativo: `#FEF2F2`

## 3. Elementos de UI (Bordas e Sombras)

- **Arredondamento (Border-Radius):** `--alfa-radius: 10px;` (Garante um aspecto "macio" e amigável).
- **Sombras (Box-Shadow):** `--alfa-shadow: 0 1px 3px rgba(0,0,0,0.07), 0 1px 2px rgba(0,0,0,0.04);` (Sombra sutil para destacar cards de KPIs e painéis sem pesar na tela).

## 4. Estilos de Gráficos (Matplotlib/Streamlit)

Qualquer gráfico inserido na aplicação deve abandonar o visual padrão da biblioteca e seguir a paleta ALFA:

- **Fundo do Gráfico (Figure & Axes):** `#FFFFFF` (Branco).
- **Bordas dos Eixos (Spines):** Remover bordas superior, direita e esquerda (`spine.set_visible(False)`). A inferior deve ser fina ou invisível.
- **Linhas de Grade (Grid):** Apenas horizontais ou ambas levemente visíveis (`#F3F4F6`, espessura `0.9`).
- **Textos e Labels do Gráfico:** Cor `#6B7280` tamanho `9` ou `10`.
- **Títulos do Gráfico:** Cor `#111827`, tamanho `12`, peso `600`.
- **Cores de Séries de Dados:**
  - Série Principal: `#4979f6` (Azul ALFA).
  - Outras Séries: `#059669` (Verde ALFA para benchmarks positivos) ou mapas de gradiente em azul (ex: `LinearSegmentedColormap` variando de `#BFDBFE` a `#1E3A8A`).

## 5. Classes CSS Padrão (Guia para HTML Injetado)

Ao criar layouts customizados via `st.markdown(..., unsafe_allow_html=True)`, utilize estas classes padronizadas (injetadas em `app.py`):

- `.alfa-kpi-card`: Para blocos de métricas. Fundo branco, borda leve, sombra sutil e raio de 10px.
- `.alfa-kpi-label`: Para o título pequeno acima ou abaixo de um valor (maiúsculas, fonte menor, texto mutado).
- `.alfa-kpi-value`: Para o número principal (fonte grande, negrito, cor escura).
- `.alfa-section-title`: Para separar subseções de tela (texto em caixa alta, espaçado, cinza escuro).

## Regra de Ouro

**Simplicidade e Consistência.** Evite adicionar novas cores "feitas na hora". O objetivo é que tudo pareça ter saído de um mesmo *Design System*. Se um botão precisa chamar atenção, ele deve usar `--alfa-accent`. Se uma métrica é boa, `--alfa-positive`. Nada além disso.
