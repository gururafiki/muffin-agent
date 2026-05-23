"""Bear Researcher node — symmetric counterpart to bull_researcher."""

from __future__ import annotations

import operator
from typing import Annotated

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from typing_extensions import TypedDict

from ....model_config import ModelConfiguration
from ....prompts import render_template
from .._debate import format_debate_history


class BearResearcherInputState(TypedDict, total=False):
    """State keys read by ``bear_researcher_node``."""

    ticker: str
    query: str
    narrative: str
    market_report: str
    fundamentals_report: str
    news_report: str
    sentiment_report: str
    investment_bull_responses: Annotated[list[str], operator.add]
    investment_bear_responses: Annotated[list[str], operator.add]


class BearResearcherOutputState(TypedDict, total=False):
    """State keys written by ``bear_researcher_node``."""

    investment_bear_responses: Annotated[list[str], operator.add]


async def bear_researcher_node(
    state: BearResearcherInputState, config: RunnableConfig
) -> BearResearcherOutputState:
    """One Bear Researcher turn. Free-form prose appended to the responses list."""
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
        ticker=state.get("ticker", ""),
        query=state.get("query"),
        narrative=state.get("narrative"),
        market_report=state.get("market_report"),
        fundamentals_report=state.get("fundamentals_report"),
        news_report=state.get("news_report"),
        sentiment_report=state.get("sentiment_report"),
        speaking_as="Bear",
        opposing_speaker="Bull",
        debate_history=format_debate_history(bulls, bears),
        opposing_last=opposing_last,
    )

    response = await llm.ainvoke(
        [
            SystemMessage(prompt),
            HumanMessage("Make your argument now."),
        ]
    )
    return {"investment_bear_responses": [str(response.content).strip()]}
