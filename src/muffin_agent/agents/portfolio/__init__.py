"""Portfolio-level decision graph nodes.

Three nodes — fully independent of ``trading_decision`` — compose into
the multi-ticker paper-trading pipeline (Phase 4.5):

* :func:`position_sizing_node` — deterministic volatility × correlation
  position-limit calculator
* :func:`ticker_decision_node` — LLM consolidator that turns a list of
  ``AnalystSignal`` (persona/specialist outputs) into one per-ticker
  recommended action + target sizing
* :func:`portfolio_reconciler_node` — hybrid deterministic + LLM step
  that resolves cross-ticker portfolio constraints (cash / margin /
  position limits) into concrete share-count orders
"""

from __future__ import annotations

from .portfolio_reconciler import (
    PortfolioReconcilerInputState,
    PortfolioReconcilerOutputState,
    PortfolioReconciliationOutput,
    portfolio_reconciler_node,
)
from .position_sizing import (
    PositionLimit,
    PositionSizingInputState,
    PositionSizingOutputState,
    compute_correlation_matrix,
    position_sizing_node,
    score_correlation_multiplier,
    score_volatility_limit,
)
from .ticker_decision import (
    TickerDecision,
    TickerDecisionInputState,
    TickerDecisionOutputState,
    ticker_decision_node,
)

__all__ = [
    "PortfolioReconcilerInputState",
    "PortfolioReconcilerOutputState",
    "PortfolioReconciliationOutput",
    "PositionLimit",
    "PositionSizingInputState",
    "PositionSizingOutputState",
    "TickerDecision",
    "TickerDecisionInputState",
    "TickerDecisionOutputState",
    "compute_correlation_matrix",
    "portfolio_reconciler_node",
    "position_sizing_node",
    "score_correlation_multiplier",
    "score_volatility_limit",
    "ticker_decision_node",
]
