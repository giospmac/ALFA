from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


DEFAULT_PORTFOLIO_COLUMNS = [
    "ticker",
    "nome",
    "tipo_ativo",
    "preco",
    "taxa",
    "porcentagem_desejada",
    "porcentagem_real",
    "qtd_acoes",
    "valor_real",
    "data_fechamento",
    "data_vencimento",
]


@dataclass(frozen=True)
class PortfolioSnapshot:
    portfolio: pd.DataFrame
    historical: pd.DataFrame

    @property
    def total_invested(self) -> float:
        if "valor_real" not in self.portfolio.columns or self.portfolio.empty:
            return 0.0
        return float(pd.to_numeric(self.portfolio["valor_real"], errors="coerce").fillna(0).sum())

    @property
    def asset_count(self) -> int:
        if self.portfolio.empty or "ticker" not in self.portfolio.columns:
            return 0
        return int(self.portfolio["ticker"].replace("", pd.NA).dropna().nunique())

    @property
    def latest_history_date(self) -> str:
        if self.historical.empty:
            return "N/A"
        return self.historical.index.max().strftime("%d/%m/%Y")


class PortfolioRepository:
    def __init__(self, base_path: str | Path | None = None):
        self.base_path = Path(base_path or Path.cwd())
        self.portfolio_file = self.base_path / "portfolio_data.csv"
        self.historical_file = self.base_path / "historico_data.csv"

    def load_snapshot(self) -> PortfolioSnapshot:
        return PortfolioSnapshot(
            portfolio=self.load_portfolio_data(),
            historical=self.load_historical_data(),
        )

    def load_portfolio_data(self) -> pd.DataFrame:
        if not self.portfolio_file.exists():
            return pd.DataFrame(columns=DEFAULT_PORTFOLIO_COLUMNS)

        try:
            df = pd.read_csv(self.portfolio_file)
        except Exception:
            return pd.DataFrame(columns=DEFAULT_PORTFOLIO_COLUMNS)

        for column in DEFAULT_PORTFOLIO_COLUMNS:
            if column not in df.columns:
                df[column] = pd.NA

        return df[DEFAULT_PORTFOLIO_COLUMNS]

    def save_portfolio_data(self, portfolio_df: pd.DataFrame) -> None:
        working_df = portfolio_df.copy()
        for column in DEFAULT_PORTFOLIO_COLUMNS:
            if column not in working_df.columns:
                working_df[column] = pd.NA

        working_df[DEFAULT_PORTFOLIO_COLUMNS].to_csv(self.portfolio_file, index=False)

    def load_historical_data(self) -> pd.DataFrame:
        if not self.historical_file.exists():
            return pd.DataFrame()

        try:
            df = pd.read_csv(self.historical_file, index_col=0)
        except Exception:
            return pd.DataFrame()

        df.index = pd.to_datetime(df.index, errors="coerce")
        df = df[df.index.notna()]

        for column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

        df = df.loc[:, ~df.columns.duplicated()]
        return df.sort_index()

    def save_historical_data(self, historical_df: pd.DataFrame) -> None:
        if historical_df.empty:
            historical_df.to_csv(self.historical_file)
            return

        working_df = historical_df.copy()
        working_df.index = pd.to_datetime(working_df.index, errors="coerce")
        working_df = working_df[working_df.index.notna()].sort_index()
        working_df.to_csv(self.historical_file)

    def load_normalized_history(self, periods: int = 180) -> pd.DataFrame:
        historical = self.load_historical_data()
        if historical.empty:
            return historical

        clean = historical.dropna(axis=1, how="all").tail(periods).ffill().dropna(axis=1, how="all")
        if clean.empty:
            return clean

        base = clean.iloc[0].replace(0, pd.NA)
        normalized = clean.divide(base).multiply(100)
        return normalized.dropna(axis=1, how="all")
