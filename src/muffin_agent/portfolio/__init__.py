"""Portfolio state + trade execution helpers for muffin's paper-trading layer.

Designed for LangGraph state: immutable Pydantic dataclasses + pure
mutation functions returning new instances.  Independent of ai-hedge-fund
and the existing ``trading_decision`` pipeline.
"""

from __future__ import annotations

from .executor import (
    ExecutedTrade,
    PortfolioOrder,
    apply_orders,
)
from .state import (
    Portfolio,
    PortfolioValue,
    Position,
    RealizedGain,
    apply_long_buy,
    apply_long_sell,
    apply_short_cover,
    apply_short_open,
    mark_to_market,
    new_portfolio,
)

__all__ = [
    "ExecutedTrade",
    "Portfolio",
    "PortfolioOrder",
    "PortfolioValue",
    "Position",
    "RealizedGain",
    "apply_long_buy",
    "apply_long_sell",
    "apply_orders",
    "apply_short_cover",
    "apply_short_open",
    "mark_to_market",
    "new_portfolio",
]
