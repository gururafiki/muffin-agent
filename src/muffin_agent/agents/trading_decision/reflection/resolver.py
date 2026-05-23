"""Reflector-resolve node — start-of-run reflection bookend.

Resolves pending decisions from prior runs (fetching realised outcomes,
generating reflections, persisting) and renders past-reflections context
for the Portfolio Manager prompt.

Per-entry failures **propagate** — no try/except. LangChain ``with_retry`` +
LangGraph node-level ``RetryPolicy`` handle transient issues. If a
deterministic failure blocks the loop, the user can clear the offending
entry manually from the per-user store.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.store.base import BaseStore
from typing_extensions import TypedDict

from ....utils.memory_config import MemoryConfiguration
from ..config import TradingDecisionConfiguration
from .memory import ReflectionMemory, render_reflections_block
from .outcomes import OutcomesFetcher, fetch_outcomes_openbb
from .reflector import reflect_on_decision

logger = logging.getLogger(__name__)


class ReflectorResolveInputState(TypedDict, total=False):
    """State keys read by ``reflector_resolve_node``."""

    analysis_context: dict[str, Any]
    decision_date: str


class ReflectorResolveOutputState(TypedDict, total=False):
    """State keys written by ``reflector_resolve_node``."""

    decision_date: str
    past_reflections: str
    resolved_decisions: list[dict[str, Any]]


def _resolve_user_id(config: RunnableConfig) -> str | None:
    """Return ``configurable.user_id`` or the debug fallback; ``None`` otherwise."""
    configurable = dict(config.get("configurable") or {})
    user_id = configurable.get("user_id")
    if isinstance(user_id, str) and user_id:
        return user_id
    try:
        debug = MemoryConfiguration.from_runnable_config(config).memory_debug_user_id
    except Exception:
        return None
    return debug or None


def _resolve_decision_date(
    config: RunnableConfig,
    state: ReflectorResolveInputState,
    cfg: TradingDecisionConfiguration,
) -> str:
    """Pick decision_date: prior state > configurable > today UTC."""
    existing = state.get("decision_date")
    if isinstance(existing, str) and existing:
        return existing
    if cfg.decision_date:
        return cfg.decision_date
    return datetime.now(UTC).strftime("%Y-%m-%d")


async def reflector_resolve_node(
    state: ReflectorResolveInputState,
    config: RunnableConfig,
    *,
    store: BaseStore | None = None,
    outcomes_fetcher: OutcomesFetcher | None = None,
) -> ReflectorResolveOutputState:
    """Start-of-run bookend: resolve pending decisions + render past-reflections.

    Always sets ``decision_date`` so ``decision_writeback_node`` uses the
    same key. When the reflection layer is unavailable (no store, no
    user_id, or ``reflection_enabled=False``), returns empty
    ``past_reflections`` / ``resolved_decisions`` and the pipeline
    continues with no learning loop.
    """
    cfg = TradingDecisionConfiguration.from_runnable_config(config)
    decision_date = _resolve_decision_date(config, state, cfg)

    base_update: ReflectorResolveOutputState = {
        "decision_date": decision_date,
        "past_reflections": "",
        "resolved_decisions": [],
    }

    if not cfg.reflection_enabled or store is None:
        return base_update

    user_id = _resolve_user_id(config)
    if user_id is None:
        return base_update

    try:
        memory = ReflectionMemory(store, user_id)
    except ValueError:
        return base_update

    fetcher: OutcomesFetcher = outcomes_fetcher or fetch_outcomes_openbb

    pending = await memory.list_pending()
    resolved_records: list[dict[str, Any]] = []
    for record in pending:
        outcome = await fetcher(
            config=config,
            ticker=record.ticker,
            decision_date=record.date,
            holding_days=cfg.reflection_holding_days,
            benchmark=cfg.reflection_benchmark,
            decision_action=record.decision.get("rating"),
        )
        if outcome is None:
            continue
        reflection = await reflect_on_decision(
            config,
            ticker=record.ticker,
            decision_date=record.date,
            decision=record.decision,
            outcome=outcome.model_dump(),
        )
        await memory.resolve(
            ticker=record.ticker,
            date=record.date,
            outcome=outcome,
            reflection=reflection,
        )
        resolved_records.append(
            {
                "ticker": record.ticker,
                "date": record.date,
                "outcome": outcome.model_dump(),
                "reflection": reflection,
            }
        )

    ticker = state["analysis_context"].get("ticker", "")
    same_ticker = await memory.list_resolved_for_ticker(
        ticker, limit=cfg.reflection_max_same_ticker
    )
    cross_ticker = await memory.list_resolved_cross_ticker(
        exclude_ticker=ticker, limit=cfg.reflection_max_cross_ticker
    )
    block = render_reflections_block(same_ticker=same_ticker, cross_ticker=cross_ticker)

    return {
        "decision_date": decision_date,
        "past_reflections": block,
        "resolved_decisions": resolved_records,
    }
