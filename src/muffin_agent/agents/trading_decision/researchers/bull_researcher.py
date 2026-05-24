"""Bull Researcher node — one LLM call, prose response."""

from __future__ import annotations

import operator
from typing import Annotated

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

    ticker: str
    query: str
    narrative: str
    market_report: str
    fundamentals_report: str
    news_report: str
    sentiment_report: str
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
    bulls = state.get("investment_bull_responses") or []
    bears = state.get("investment_bear_responses") or []
    opposing_last: str = bears[-1] if bears else ""

    llm = ModelConfiguration.get_chat_model_for_role(config, "reasoner")

    prompt = render_template(
        "trading_decision/researchers/bull.jinja",
        ticker=state.get("ticker", ""),
        query=state.get("query"),
        narrative=state.get("narrative"),
        market_report=state.get("market_report"),
        fundamentals_report=state.get("fundamentals_report"),
        news_report=state.get("news_report"),
        sentiment_report=state.get("sentiment_report"),
        speaking_as="Bull",
        opposing_speaker="Bear",
        debate_history=format_debate_history(bulls, bears),
        opposing_last=opposing_last,
    )

    response = await llm.ainvoke(
        [
            SystemMessage(prompt),
            HumanMessage("Make your argument now."),
        ]
    )
    return {"investment_bull_responses": [str(response.content).strip()]}
