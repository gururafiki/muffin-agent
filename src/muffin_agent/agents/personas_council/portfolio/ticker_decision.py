"""Per-ticker decision node — turn ``AnalystSignal`` list into one recommendation.

LLM call that consolidates persona / specialist signals for a single ticker
into one ``TickerDecision``.  Used by the multi-ticker portfolio decision
graph (Phase 4.5) as the fan-out worker.

This is **independent of trading_decision** — it consumes the council
output, not the Judge/Trader artefacts.
"""

from __future__ import annotations

import logging
from typing import Any, Literal, cast

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from ....model_config import ModelConfiguration
from ....prompts import render_template
from ..schemas import InvestmentSignal

logger = logging.getLogger(__name__)

RecommendedAction = Literal["buy", "sell", "short", "cover", "hold"]


class TickerDecision(BaseModel):
    """Per-ticker action recommendation produced by the LLM consolidator."""

    ticker: str
    recommended_action: RecommendedAction
    target_pct_of_nav: float = Field(ge=0.0, le=1.0, default=0.0)
    """Suggested position size as a fraction of NAV.  ``0`` for ``hold``."""

    rating: InvestmentSignal
    """5-tier consensus rating used to derive the action."""

    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
    signals_summary: dict[str, list[str]] = Field(default_factory=dict)
    """Rating → list of agent_ids that voted that way (from upstream)."""


class TickerDecisionInputState(TypedDict, total=False):
    """State keys read by ``ticker_decision_node``."""

    ticker: str
    query: str
    persona_signals: list[dict[str, Any]]
    council_synthesis: dict[str, Any]


class TickerDecisionOutputState(TypedDict, total=False):
    """State keys written by ``ticker_decision_node``."""

    ticker_decision: dict[str, Any]


async def ticker_decision_node(
    state: TickerDecisionInputState, config: RunnableConfig
) -> TickerDecisionOutputState:
    """Render a single ``TickerDecision`` for a ticker from upstream signals.

    Inputs in state:
        ticker: equity ticker symbol
        query: optional investment mandate
        persona_signals: list of ``AnalystSignal.model_dump()`` dicts
        council_synthesis: optional ``CouncilSynthesisOutput.model_dump()``

    Output: ``state["ticker_decision"] = TickerDecision.model_dump()``.
    """
    ticker = state.get("ticker", "")
    query = state.get("query")
    persona_signals = state.get("persona_signals") or []
    council = state.get("council_synthesis") or {}

    # If no signals at all, default to hold without an LLM call.
    if not persona_signals and not council:
        fallback = TickerDecision(
            ticker=ticker,
            recommended_action="hold",
            target_pct_of_nav=0.0,
            rating="hold",
            confidence=0.0,
            reasoning="No upstream signals available",
            signals_summary={},
        )
        return {"ticker_decision": fallback.model_dump()}

    llm = ModelConfiguration.get_chat_model_for_role(
        config, "reasoner", schema=TickerDecision
    )
    prompt = render_template(
        "portfolio/ticker_decision.jinja",
        ticker=ticker,
        query=query,
        persona_signals=persona_signals,
        council_synthesis=council,
    )
    result = cast(
        TickerDecision,
        await llm.ainvoke(
            [
                SystemMessage(prompt),
                HumanMessage("Render the ticker decision now."),
            ]
        ),
    )
    return {"ticker_decision": result.model_dump()}
