"""Council judge — synthesises 13 persona signals into one consensus rating.

Single LLM call against ``CouncilSynthesisOutput`` Pydantic schema.
The judge consumes ``state["persona_signals"]`` (list of dicts from the
13 fan-out workers) and produces:

* ``consensus_rating`` — 5-tier verdict, LLM-mediated synthesis
* ``weighted_confidence`` — overall confidence
* ``vote_breakdown`` — rating → list of persona slugs that voted that way
* ``bull_case_synthesis`` / ``bear_case_synthesis``
* ``dissent_summary``, ``key_uncertainties``, ``reasoning``
"""

from __future__ import annotations

import logging
from typing import Any, cast

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from ...model_config import ModelConfiguration
from ...prompts import render_template
from .schemas import InvestmentSignal

logger = logging.getLogger(__name__)


class CouncilSynthesisOutput(BaseModel):
    """Council judge's structured verdict."""

    ticker: str
    consensus_rating: InvestmentSignal
    weighted_confidence: float = Field(ge=0.0, le=1.0)
    vote_breakdown: dict[str, list[str]] = Field(default_factory=dict)
    """Rating → list of persona slugs that voted that way (5 keys max)."""

    bull_case_synthesis: str
    """Strongest bullish argument distilled across all bullish personas."""

    bear_case_synthesis: str
    """Strongest bearish argument distilled across all bearish personas."""

    dissent_summary: str
    """Narrative on notable dissenters — who disagreed with the consensus and why."""

    key_uncertainties: list[str] = Field(default_factory=list)
    """Open questions or data gaps that would change the verdict."""

    reasoning: str
    """1–3 sentence overall justification for the consensus rating."""


class CouncilJudgeInputState(TypedDict, total=False):
    """State keys read by the council judge."""

    ticker: str
    query: str
    persona_signals: list[dict[str, Any]]


class CouncilJudgeOutputState(TypedDict, total=False):
    """State keys written by the council judge."""

    council_synthesis: dict[str, Any]


async def council_judge_node(
    state: CouncilJudgeInputState, config: RunnableConfig
) -> CouncilJudgeOutputState:
    """Synthesise persona signals into a single ``CouncilSynthesisOutput``.

    Reads ``state["persona_signals"]`` (accumulated by the council graph's
    parallel fan-out), pre-computes a deterministic vote breakdown for
    the LLM's reference, then makes ONE LLM call.
    """
    ticker = state.get("ticker", "")
    signals = state.get("persona_signals") or []

    # Deterministic pre-aggregation for the LLM prompt
    vote_breakdown: dict[str, list[str]] = {
        "strong_sell": [],
        "sell": [],
        "hold": [],
        "buy": [],
        "strong_buy": [],
    }
    confidences: dict[str, float] = {}
    for sig in signals:
        rating = sig.get("signal")
        agent_id = sig.get("agent_id", "unknown")
        if rating in vote_breakdown:
            vote_breakdown[rating].append(agent_id)
        confidences[agent_id] = float(sig.get("confidence") or 0.0)

    if not signals:
        # Defensive fallback when no personas reported
        fallback = CouncilSynthesisOutput(
            ticker=ticker,
            consensus_rating="hold",
            weighted_confidence=0.0,
            vote_breakdown=vote_breakdown,
            bull_case_synthesis="No persona signals available.",
            bear_case_synthesis="No persona signals available.",
            dissent_summary="No council members reported.",
            key_uncertainties=[
                "No persona signals available — data collection failed."
            ],
            reasoning="Defaulting to hold; no council input available.",
        )
        return {"council_synthesis": fallback.model_dump()}

    llm = ModelConfiguration.get_chat_model_for_role(
        config, "reasoner", schema=CouncilSynthesisOutput
    )
    prompt = render_template(
        "personas_council/council_judge.jinja",
        ticker=ticker,
        query=state.get("query"),
        signals=signals,
        vote_breakdown=vote_breakdown,
        confidences=confidences,
        persona_count=len(signals),
    )
    result = cast(
        CouncilSynthesisOutput,
        await llm.ainvoke(
            [
                SystemMessage(prompt),
                HumanMessage("Render the council's consensus verdict now."),
            ]
        ),
    )
    return {"council_synthesis": result.model_dump()}
