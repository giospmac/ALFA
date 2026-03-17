import json
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from services.portfolio_analytics import MarkowitzResult

columns = ["AAPL", "MSFT", "GOOGL", "NVDA", "AMZN"]
results = np.zeros((3 + len(columns), 10000))
rng = np.random.default_rng()
for i in range(10000):
    w = rng.random(len(columns))
    w /= w.sum()
    results[0, i] = rng.normal(0.1, 0.05)
    results[1, i] = rng.normal(0.2, 0.05)
    results[2, i] = results[0, i] / results[1, i]
    results[3:, i] = w

portfolios = pd.DataFrame(results.T, columns=["retorno", "volatilidade", "sharpe", *columns])

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=portfolios["volatilidade"] * 100,
    y=portfolios["retorno"] * 100,
    mode="markers",
    marker=dict(
        color=portfolios["sharpe"],
        colorscale="Viridis",
        showscale=True,
        size=6,
        colorbar=dict(title="Sharpe")
    ),
    text=[
        "<br>".join([f"{col}: {row[col] * 100:.1f}%" for col in portfolios.columns[3:]])
        for _, row in portfolios.iterrows()
    ],
    hoverinfo="text",
    name="Portfolios"
))

j = fig.to_json()
print("Markowitz Fig size in bytes:", len(j))

# Monte Carlo figure size
sims = rng.normal(0, 0.01, size=(20, 10000))
mc_returns = np.exp(sims.sum(axis=0)) - 1
fig2 = go.Figure()
fig2.add_trace(go.Histogram(
    x=mc_returns * 100,
    nbinsx=50
))
print("Monte Carlo Fig size in bytes:", len(fig2.to_json()))

