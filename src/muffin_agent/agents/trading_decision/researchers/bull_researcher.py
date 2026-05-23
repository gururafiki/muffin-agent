"""Bull Researcher node — one LLM call, prose response."""

from __future__ import annotations

import operator
from typing import Annotated, Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from typing_extensions import TypedDict

from ....model_config import ModelConfiguration
from ....prompts import render_template
from .._debate import format_debate_history


class BullResearcherInputState(TypedDict, total=False):
    """State keys this node reads.

    Any graph that satisfies this shape can reuse ``bull_researcher_node``.
    """

    analysis_context: dict[str, Any]
    investment_bull_responses: Annotated[list[str], operator.add]
    investment_bear_responses: Annotated[list[str], operator.add]


class BullResearcherOutputState(TypedDict, total=False):
    """State keys this node writes."""

    investment_bull_responses: Annotated[list[str], operator.add]


async def bull_researcher_node(
    state: BullResearcherInputState, config: RunnableConfig
) -> BullResearcherOutputState:
    """One Bull Researcher turn. Free-form prose appended to the responses list.

    Routing is decided by the graph-level ``_route_investment_debate`` function;
    this node only updates state.
    """
    analysis_context = state["analysis_context"]
    bulls = state.get("investment_bull_responses") or []
    bears = state.get("investment_bear_responses") or []
    opposing_last: str = bears[-1] if bears else ""

    cfg = ModelConfiguration.from_runnable_config(config)
    primary, *fallbacks = cfg.get_llm_for_role("reasoner")
    llm = (primary.with_fallbacks(fallbacks) if fallbacks else primary).with_retry(
        stop_after_attempt=3, wait_exponential_jitter=True
    )

    prompt = render_template(
        "trading_decision/researchers/bull.jinja",
        ticker=analysis_context.get("ticker", ""),
        query=analysis_context.get("query"),
        speaking_as="Bull",
        opposing_speaker="Bear",
        debate_history=format_debate_history(bulls, bears),
        opposing_last=opposing_last,
        market_regime=analysis_context.get("market_regime"),
        sector_view=analysis_context.get("sector_view"),
        company_analysis=analysis_context.get("company_analysis"),
        forecast=analysis_context.get("forecast"),
        risk_assessment=analysis_context.get("risk_assessment"),
        valuation=analysis_context.get("valuation"),
        narrative=analysis_context.get("narrative"),
        additional_context=analysis_context.get("additional_context") or {},
    )

    response = await llm.ainvoke(
        [
            SystemMessage(prompt),
            HumanMessage("Make your argument now."),
        ]
    )
    return {"investment_bull_responses": [str(response.content).strip()]}
