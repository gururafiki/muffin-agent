"""Decision-writeback node — end-of-run reflection bookend.

Persists the current ``portfolio_decision`` as a ``pending`` record under
the per-user reflection namespace. Degrades silently when reflection infra
is unavailable; skips errored / missing decisions.
"""

from __future__ import annotations

from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.store.base import BaseStore
from typing_extensions import TypedDict

from .memory import try_build_reflection_memory


class DecisionWritebackInputState(TypedDict, total=False):
    """State keys read by :func:`decision_writeback_node`."""

    ticker: str
    portfolio_decision: dict[str, Any]
    decision_date: str


class DecisionWritebackOutputState(TypedDict, total=False):
    """State keys written by :func:`decision_writeback_node`.

    Empty — this node only writes side-effects (store persistence).
    """


async def decision_writeback_node(
    state: DecisionWritebackInputState,
    config: RunnableConfig,
    *,
    store: BaseStore | None = None,
) -> DecisionWritebackOutputState:
    """Persist this run's ``PortfolioDecisionOutput`` as a pending entry.

    No-ops when reflection is disabled, the store is not wired, no
    ``user_id`` is resolvable, or the decision payload is missing /
    carries an ``error`` key.
    """
    memory = try_build_reflection_memory(config, store)
    if memory is None:
        return {}

    decision: Any = state.get("portfolio_decision")
    if not isinstance(decision, dict) or "error" in decision:
        return {}

    ticker = state.get("ticker") or ""
    decision_date = state.get("decision_date") or ""
    if not ticker or not decision_date:
        return {}

    await memory.write_pending(ticker=ticker, date=decision_date, decision=decision)
    return {}
