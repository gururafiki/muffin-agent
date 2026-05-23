"""Decision-writeback node — end-of-run reflection bookend.

Persists the current ``portfolio_decision`` as a ``pending`` record under
the per-user reflection namespace. Degrades silently when reflection infra
is unavailable; skips errored / missing decisions.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.store.base import BaseStore
from typing_extensions import TypedDict

from ....utils.memory_config import MemoryConfiguration
from ..config import TradingDecisionConfiguration
from .memory import ReflectionMemory

logger = logging.getLogger(__name__)


class DecisionWritebackInputState(TypedDict, total=False):
    """State keys read by ``decision_writeback_node``."""

    analysis_context: dict[str, Any]
    portfolio_decision: dict[str, Any]
    decision_date: str


class DecisionWritebackOutputState(TypedDict, total=False):
    """State keys written by ``decision_writeback_node``.

    Empty — this node only writes side-effects (store persistence).
    """


def _resolve_user_id(config: RunnableConfig) -> str | None:
    configurable = dict(config.get("configurable") or {})
    user_id = configurable.get("user_id")
    if isinstance(user_id, str) and user_id:
        return user_id
    try:
        debug = MemoryConfiguration.from_runnable_config(config).memory_debug_user_id
    except Exception:
        return None
    return debug or None


async def decision_writeback_node(
    state: DecisionWritebackInputState,
    config: RunnableConfig,
    *,
    store: BaseStore | None = None,
) -> DecisionWritebackOutputState:
    """Persist this run's decision as a pending entry for future reflection.

    No-ops when:
      * Reflection is disabled.
      * The store is not wired.
      * No ``user_id`` is resolvable.
      * The decision payload is missing or carries an ``error`` key.
    """
    cfg = TradingDecisionConfiguration.from_runnable_config(config)
    if not cfg.reflection_enabled or store is None:
        return {}

    user_id = _resolve_user_id(config)
    if user_id is None:
        return {}

    decision: Any = state.get("portfolio_decision")
    if not isinstance(decision, dict) or "error" in decision:
        return {}

    ticker = state["analysis_context"].get("ticker", "")
    decision_date = state.get("decision_date") or ""
    if not ticker or not decision_date:
        return {}

    try:
        memory = ReflectionMemory(store, user_id)
    except ValueError:
        return {}

    await memory.write_pending(ticker=ticker, date=decision_date, decision=decision)
    return {}
