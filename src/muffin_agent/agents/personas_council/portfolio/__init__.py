"""Paper-trading layer: portfolio state + risk/decision nodes + multi-ticker graph.

Immutable Pydantic state (``state.py``) + pure mutation helpers, the
deterministic risk-manager (position-limit) node, the LLM ticker-decision +
reconciler nodes, and the multi-ticker ``portfolio_decision`` graph.
Independent of the ``trading_decision`` pipeline.
"""

from __future__ import annotations

from .executor import (
    ExecutedTrade,
    PortfolioOrder,
    apply_orders,
)
from .portfolio_decision import build_portfolio_decision_graph
from .portfolio_reconciler import (
    PortfolioReconcilerInputState,
    PortfolioReconcilerOutputState,
    PortfolioReconciliationOutput,
    portfolio_reconciler_node,
)
from .risk_manager import (
    PositionLimit,
    PositionSizingInputState,
    PositionSizingOutputState,
    compute_correlation_matrix,
    risk_management_node,
    score_correlation_multiplier,
    score_volatility_limit,
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
from .ticker_decision import (
    TickerDecision,
    TickerDecisionInputState,
    TickerDecisionOutputState,
    ticker_decision_node,
)

__all__ = [
    "ExecutedTrade",
    "Portfolio",
    "PortfolioOrder",
    "PortfolioReconcilerInputState",
    "PortfolioReconcilerOutputState",
    "PortfolioReconciliationOutput",
    "PortfolioValue",
    "Position",
    "PositionLimit",
    "PositionSizingInputState",
    "PositionSizingOutputState",
    "RealizedGain",
    "TickerDecision",
    "TickerDecisionInputState",
    "TickerDecisionOutputState",
    "apply_long_buy",
    "apply_long_sell",
    "apply_orders",
    "apply_short_cover",
    "apply_short_open",
    "build_portfolio_decision_graph",
    "compute_correlation_matrix",
    "mark_to_market",
    "new_portfolio",
    "portfolio_reconciler_node",
    "risk_management_node",
    "score_correlation_multiplier",
    "score_volatility_limit",
    "ticker_decision_node",
]
