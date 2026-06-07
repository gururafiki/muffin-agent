"""Walk-forward backtester for the persona council + paper-trading pipeline.

Two modes — ``full`` (entire portfolio decision graph per rebalance)
and ``signals`` (council only, no execution).  Independent of
``trading_decision``; reuses muffin's portfolio / executor / council
infrastructure.
"""

from __future__ import annotations

from .engine import (
    BacktestEngine,
    BacktestMode,
    BacktestResults,
    RebalanceSnapshot,
    synthetic_price_provider,
)
from .metrics import (
    DEFAULT_RISK_FREE_RATE,
    compute_benchmark_comparison,
    compute_max_drawdown,
    compute_returns_from_equity,
    compute_sharpe,
    compute_sortino,
    compute_total_return,
)

__all__ = [
    "DEFAULT_RISK_FREE_RATE",
    "BacktestEngine",
    "BacktestMode",
    "BacktestResults",
    "RebalanceSnapshot",
    "compute_benchmark_comparison",
    "compute_max_drawdown",
    "compute_returns_from_equity",
    "compute_sharpe",
    "compute_sortino",
    "compute_total_return",
    "synthetic_price_provider",
]
