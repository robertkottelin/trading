"""Execution module — dYdX v4 trade execution layer.

Provides paper and live trading executors with pre-trade risk validation.
"""

from execution.risk_manager import RiskManager
from execution.paper_executor import PaperExecutor

__all__ = ["RiskManager", "PaperExecutor"]

# Live executor imports require dydx-v4-client SDK — import explicitly:
#   from execution.dydx_client import DydxClient
#   from execution.dydx_executor import DydxExecutor
