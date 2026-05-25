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

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.store.base import BaseStore
from typing_extensions import TypedDict

from ....model_config import ModelConfiguration
from ....prompts import render_template
from ..config import TradingDecisionConfiguration
from ..schemas import DecisionRecord
from ..tools import OutcomesFetcher, fetch_decision_outcome
from .memory import (
    ReflectionMemory,
    render_reflections_block,
    try_build_reflection_memory,
)

logger = logging.getLogger(__name__)


class ReflectorResolveInputState(TypedDict, total=False):
    """State keys read by :func:`reflector_resolve_node`."""

    ticker: str
    decision_date: str


class ReflectorResolveOutputState(TypedDict, total=False):
    """State keys written by :func:`reflector_resolve_node`."""

    decision_date: str
    past_reflections: str
    resolved_decisions: list[dict[str, Any]]


def _resolve_decision_date(
    state: ReflectorResolveInputState,
    cfg: TradingDecisionConfiguration,
) -> str:
    """Pick ``decision_date``: prior state > configurable > today UTC."""
    existing = state.get("decision_date")
    if isinstance(existing, str) and existing:
        return existing
    if cfg.decision_date:
        return cfg.decision_date
    return datetime.now(UTC).strftime("%Y-%m-%d")


async def _reflect_on_decision(
    config: RunnableConfig,
    *,
    ticker: str,
    decision_date: str,
    decision: dict[str, Any],
    outcome: dict[str, Any],
) -> str:
    """Generate a 2-4 sentence reflection on a (decision, outcome) pair.

    Inlined here (was ``reflection/reflector.py``) — single caller, so the
    separate-module indirection didn't earn its keep at ~30 lines of body.
    """
    llm = ModelConfiguration.get_chat_model_for_role(config, "reasoner")
    prompt = render_template(
        "trading_decision/reflection/reflector.jinja",
        ticker=ticker,
        decision_date=decision_date,
        decision=decision,
        outcome=outcome,
    )
    response = await llm.ainvoke(
        [SystemMessage(prompt), HumanMessage("Write your reflection now.")]
    )
    return str(response.content).strip()


async def _resolve_one_pending(
    config: RunnableConfig,
    *,
    record: DecisionRecord,
    cfg: TradingDecisionConfiguration,
    fetcher: OutcomesFetcher,
    memory: ReflectionMemory,
) -> dict[str, Any] | None:
    """Fetch outcome + generate reflection + persist for a single pending record.

    Returns the resolved-record dict for observability, or ``None`` if the
    outcome isn't available yet (leaves the record pending for next run).
    """
    outcome = await fetcher(
        config=config,
        ticker=record.ticker,
        decision_date=record.date,
        holding_days=cfg.reflection_holding_days,
        benchmark=cfg.reflection_benchmark,
        decision_action=record.decision.get("rating"),
    )
    if outcome is None:
        return None
    reflection = await _reflect_on_decision(
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
    return {
        "ticker": record.ticker,
        "date": record.date,
        "outcome": outcome.model_dump(),
        "reflection": reflection,
    }


async def reflector_resolve_node(
    state: ReflectorResolveInputState,
    config: RunnableConfig,
    *,
    store: BaseStore | None = None,
    outcomes_fetcher: OutcomesFetcher | None = None,
) -> ReflectorResolveOutputState:
    """Start-of-run bookend: resolve pending decisions + render past-reflections.

    Always sets ``decision_date`` so ``decision_writeback_node`` uses the
    same key. When the reflection layer is unavailable, returns empty
    ``past_reflections`` / ``resolved_decisions`` and the pipeline continues.
    """
    cfg = TradingDecisionConfiguration.from_runnable_config(config)
    decision_date = _resolve_decision_date(state, cfg)

    memory = try_build_reflection_memory(config, store)
    if memory is None:
        return {
            "decision_date": decision_date,
            "past_reflections": "",
            "resolved_decisions": [],
        }

    fetcher: OutcomesFetcher = outcomes_fetcher or fetch_decision_outcome

    resolved_records: list[dict[str, Any]] = []
    for record in await memory.list_pending():
        resolved = await _resolve_one_pending(
            config, record=record, cfg=cfg, fetcher=fetcher, memory=memory
        )
        if resolved is not None:
            resolved_records.append(resolved)

    ticker = state.get("ticker") or ""
    same_ticker = await memory.list_resolved_for_ticker(
        ticker, limit=cfg.reflection_max_same_ticker
    )
    cross_ticker = await memory.list_resolved_cross_ticker(
        exclude_ticker=ticker, limit=cfg.reflection_max_cross_ticker
    )
    return {
        "decision_date": decision_date,
        "past_reflections": render_reflections_block(
            same_ticker=same_ticker, cross_ticker=cross_ticker
        ),
        "resolved_decisions": resolved_records,
    }
