"""Shared :class:`OutcomesFetcher` Protocol + default implementation.

This module is the **canonical import location** for the outcomes-fetching
abstraction.  Trading-decision and the new backtester both depend on
it; keeping the Protocol here (rather than buried inside
``trading_decision.tools``) makes it explicit that this is a
cross-pipeline utility.

For backward compatibility the original symbols also remain importable
from ``muffin_agent.agents.trading_decision.tools``; this module simply
re-exports them under a more discoverable path.
"""

from __future__ import annotations

from ..agents.trading_decision.schemas import Outcome
from ..agents.trading_decision.tools import (
    OutcomesFetcher,
    fetch_decision_outcome,
)

__all__ = [
    "Outcome",
    "OutcomesFetcher",
    "fetch_decision_outcome",
]
