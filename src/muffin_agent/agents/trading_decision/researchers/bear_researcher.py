"""Bear Researcher node — symmetric counterpart to bull_researcher."""

from __future__ import annotations

import operator
from typing import Annotated, Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from typing_extensions import TypedDict

from ....model_config import ModelConfiguration
from ....prompts import render_template
from .._debate import format_debate_history


class BearResearcherInputState(TypedDict, total=False):
    """State keys read by ``bear_researcher_node``."""

    analysis_context: dict[str, Any]
    investment_bull_responses: Annotated[list[str], operator.add]
    investment_bear_responses: Annotated[list[str], operator.add]


class BearResearcherOutputState(TypedDict, total=False):
    """State keys written by ``bear_researcher_node``."""

    investment_bear_responses: Annotated[list[str], operator.add]


async def bear_researcher_node(
    state: BearResearcherInputState, config: RunnableConfig
) -> BearResearcherOutputState:
    """One Bear Researcher turn. Free-form prose appended to the responses list."""
    analysis_context = state["analysis_context"]
    bulls = state.get("investment_bull_responses") or []
    bears = state.get("investment_bear_responses") or []
    opposing_last: str = bulls[-1] if bulls else ""

    cfg = ModelConfiguration.from_runnable_config(config)
    primary, *fallbacks = cfg.get_llm_for_role("reasoner")
    llm = (primary.with_fallbacks(fallbacks) if fallbacks else primary).with_retry(
        stop_after_attempt=3, wait_exponential_jitter=True
    )

    prompt = render_template(
        "trading_decision/researchers/bear.jinja",
        ticker=analysis_context.get("ticker", ""),
        query=analysis_context.get("query"),
        speaking_as="Bear",
        opposing_speaker="Bull",
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
    return {"investment_bear_responses": [str(response.content).strip()]}
