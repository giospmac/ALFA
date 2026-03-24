"""CRUD repository for transactions and cash ledger CSV files."""
from __future__ import annotations

import uuid
from datetime import datetime
from pathlib import Path

import pandas as pd


# ------------------------------------------------------------------
# Schemas
# ------------------------------------------------------------------

TRANSACTIONS_COLUMNS = [
    "trade_id", "timestamp", "ticker", "asset_type", "side",
    "quantity", "price", "gross_value", "fees", "net_value",
    "strategy_tag", "notes",
]

CASH_LEDGER_COLUMNS = [
    "event_id", "timestamp", "event_type", "amount", "reference_id", "notes",
]

NAV_HISTORY_COLUMNS = [
    "date", "pl_total", "cash_balance", "units_outstanding",
    "nav_per_unit", "net_flow", "gross_return", "net_return",
]


# ------------------------------------------------------------------
# Repository
# ------------------------------------------------------------------

class TransactionsRepository:
    """Manages ``transactions_data.csv``, ``cash_ledger.csv`` and ``fund_nav_history.csv``."""

    def __init__(self, base_path: str | Path | None = None) -> None:
        self.base_path = Path(base_path or Path.cwd())
        self.transactions_file = self.base_path / "transactions_data.csv"
        self.cash_ledger_file = self.base_path / "cash_ledger.csv"
        self.nav_history_file = self.base_path / "fund_nav_history.csv"

    # ------------------------------------------------------------------
    # Transactions
    # ------------------------------------------------------------------

    def load_transactions(self) -> pd.DataFrame:
        if not self.transactions_file.exists():
            return pd.DataFrame(columns=TRANSACTIONS_COLUMNS)
        try:
            df = pd.read_csv(self.transactions_file)
        except Exception:
            return pd.DataFrame(columns=TRANSACTIONS_COLUMNS)
        for c in TRANSACTIONS_COLUMNS:
            if c not in df.columns:
                df[c] = pd.NA
        return df[TRANSACTIONS_COLUMNS]

    def save_transactions(self, df: pd.DataFrame) -> None:
        for c in TRANSACTIONS_COLUMNS:
            if c not in df.columns:
                df[c] = pd.NA
        df[TRANSACTIONS_COLUMNS].to_csv(self.transactions_file, index=False)

    def add_transaction(
        self,
        ticker: str,
        side: str,
        quantity: float,
        price: float,
        asset_type: str = "acao",
        fees: float = 0.0,
        strategy_tag: str = "",
        notes: str = "",
    ) -> pd.DataFrame:
        gross = quantity * price
        net = gross + fees if side in ("BUY", "SUBSCRIPTION") else gross - fees
        row = {
            "trade_id": str(uuid.uuid4())[:8],
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "ticker": ticker,
            "asset_type": asset_type,
            "side": side,
            "quantity": quantity,
            "price": price,
            "gross_value": gross,
            "fees": fees,
            "net_value": net,
            "strategy_tag": strategy_tag,
            "notes": notes,
        }
        df = self.load_transactions()
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        self.save_transactions(df)
        return df

    # ------------------------------------------------------------------
    # Cash Ledger
    # ------------------------------------------------------------------

    def load_cash_ledger(self) -> pd.DataFrame:
        if not self.cash_ledger_file.exists():
            return pd.DataFrame(columns=CASH_LEDGER_COLUMNS)
        try:
            df = pd.read_csv(self.cash_ledger_file)
        except Exception:
            return pd.DataFrame(columns=CASH_LEDGER_COLUMNS)
        for c in CASH_LEDGER_COLUMNS:
            if c not in df.columns:
                df[c] = pd.NA
        return df[CASH_LEDGER_COLUMNS]

    def save_cash_ledger(self, df: pd.DataFrame) -> None:
        for c in CASH_LEDGER_COLUMNS:
            if c not in df.columns:
                df[c] = pd.NA
        df[CASH_LEDGER_COLUMNS].to_csv(self.cash_ledger_file, index=False)

    def add_cash_event(
        self,
        event_type: str,
        amount: float,
        reference_id: str = "",
        notes: str = "",
    ) -> pd.DataFrame:
        row = {
            "event_id": str(uuid.uuid4())[:8],
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "event_type": event_type,
            "amount": amount,
            "reference_id": reference_id,
            "notes": notes,
        }
        df = self.load_cash_ledger()
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        self.save_cash_ledger(df)
        return df

    # ------------------------------------------------------------------
    # NAV History
    # ------------------------------------------------------------------

    def load_nav_history(self) -> pd.DataFrame:
        if not self.nav_history_file.exists():
            return pd.DataFrame(columns=NAV_HISTORY_COLUMNS)
        try:
            df = pd.read_csv(self.nav_history_file)
        except Exception:
            return pd.DataFrame(columns=NAV_HISTORY_COLUMNS)
        for c in NAV_HISTORY_COLUMNS:
            if c not in df.columns:
                df[c] = pd.NA
        return df[NAV_HISTORY_COLUMNS]

    def save_nav_history(self, df: pd.DataFrame) -> None:
        for c in NAV_HISTORY_COLUMNS:
            if c not in df.columns:
                df[c] = pd.NA
        df[NAV_HISTORY_COLUMNS].to_csv(self.nav_history_file, index=False)

    def append_nav_record(self, record: dict) -> pd.DataFrame:
        df = self.load_nav_history()
        df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
        self.save_nav_history(df)
        return df
