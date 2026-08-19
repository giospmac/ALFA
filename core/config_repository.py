from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class ModelConfig:
    """Centralised model premisses persisted as JSON."""

    risk_free_rate: float = 0.1075       # Rf annualised (default: CDI 10.75%)
    emrp: float = 0.06                   # Equity Market Risk Premium (expected)
    delta: float = 2.5                    # Risk aversion (Black-Litterman)
    tau: float = 0.05                     # Uncertainty scalar (Black-Litterman)
    max_weight: float = 1.0              # Max weight per asset in optimisation
    allow_short: bool = False            # Allow short-selling in optimisation


class ConfigRepository:
    """Load / save ``ModelConfig`` to a JSON file at *base_path*/config.json."""

    _DEFAULT_FILENAME = "config.json"

    def __init__(self, base_path: str | Path | None = None) -> None:
        self.base_path = Path(base_path or Path.cwd())
        self.config_file = self.base_path / self._DEFAULT_FILENAME

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self) -> ModelConfig:
        if not self.config_file.exists():
            return ModelConfig()
        try:
            with open(self.config_file, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            return ModelConfig(**{k: v for k, v in data.items() if k in ModelConfig.__dataclass_fields__})
        except Exception:
            return ModelConfig()

    def save(self, config: ModelConfig) -> None:
        with open(self.config_file, "w", encoding="utf-8") as fh:
            json.dump(asdict(config), fh, indent=2, ensure_ascii=False)

    def update(self, **kwargs) -> ModelConfig:
        """Load, patch and persist in one call."""
        config = self.load()
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
        self.save(config)
        return config
