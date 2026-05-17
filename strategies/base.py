"""Base class and data structures for conventional trading strategies."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd


@dataclass
class StrategySignal:
    """Output from a strategy evaluation."""
    direction: str  # "LONG", "SHORT", or "INACTIVE"
    confidence: float  # 0.0 - 1.0
    explanation: str  # Human-readable reason
    details: dict = field(default_factory=dict)  # Key metrics


class BaseStrategy(ABC):
    """Abstract base for all conventional strategies."""

    name: str = ""
    description: str = ""
    data_files: list[str] = []  # Required CSV filenames

    def __init__(self):
        """Load tuned parameter overrides from config/strategy_params.yaml if present."""
        params_path = Path("config/strategy_params.yaml")
        if params_path.exists():
            try:
                import yaml
                all_params = yaml.safe_load(params_path.read_text()) or {}
                key = type(self).__name__.lower()
                overrides = all_params.get(key, {})
                for k, v in overrides.items():
                    if hasattr(self, k):
                        setattr(self, k, v)
            except Exception:
                pass  # Never break the pipeline over a bad params file

    def load_data(self, data_dir: str) -> dict[str, pd.DataFrame]:
        """Load required CSVs from a data directory."""
        data = {}
        for fname in self.data_files:
            path = Path(data_dir) / fname
            if path.exists():
                data[fname] = pd.read_csv(path)
            else:
                data[fname] = pd.DataFrame()
        return data

    @abstractmethod
    def compute_signal(self, data: dict[str, pd.DataFrame]) -> StrategySignal:
        """Compute the current signal from latest data (for live pipeline)."""
        ...

    @abstractmethod
    def compute_signal_series(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Compute full signal time series for backtesting.

        Returns DataFrame with columns: date, signal (1=LONG, -1=SHORT, 0=INACTIVE),
        confidence, and strategy-specific detail columns.
        """
        ...
