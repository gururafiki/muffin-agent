"""Aggressive Risk Debator node."""

from __future__ import annotations

import operator
from typing import Annotated, Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from typing_extensions import TypedDict

from ....model_config import ModelConfiguration
from ....prompts import render_template
from .._debate import format_risk_history


class AggressiveDebatorInputState(TypedDict, total=False):
    """State keys read by ``aggressive_debator_node``."""

    ticker: str
    query: str
    narrative: str
    market_report: str
    fundamentals_report: str
    news_report: str
    sentiment_report: str
    investment_judge: dict[str, Any]
    trader: dict[str, Any]
    risk_aggressive_responses: Annotated[list[str], operator.add]
    risk_conservative_responses: Annotated[list[str], operator.add]
    risk_neutral_responses: Annotated[list[str], operator.add]


class AggressiveDebatorOutputState(TypedDict, total=False):
    """State keys written by ``aggressive_debator_node``."""

    risk_aggressive_responses: Annotated[list[str], operator.add]


async def aggressive_debator_node(
    state: AggressiveDebatorInputState, config: RunnableConfig
) -> AggressiveDebatorOutputState:
    """One Aggressive Risk Debator turn. Prose appended to responses list."""
    aggressives = state.get("risk_aggressive_responses") or []
    conservatives = state.get("risk_conservative_responses") or []
    neutrals = state.get("risk_neutral_responses") or []

    llm = ModelConfiguration.get_chat_model_for_role(config, "reasoner")

    prompt = render_template(
        "trading_decision/risk_debate/aggressive.jinja",
        ticker=state.get("ticker", ""),
        query=state.get("query"),
        narrative=state.get("narrative"),
        market_report=state.get("market_report"),
        fundamentals_report=state.get("fundamentals_report"),
        news_report=state.get("news_report"),
        sentiment_report=state.get("sentiment_report"),
        investment_judge=state["investment_judge"],
        trader=state["trader"],
        risk_debate_history=format_risk_history(aggressives, conservatives, neutrals),
    )

    response = await llm.ainvoke(
        [
            SystemMessage(prompt),
            HumanMessage("Make your argument now."),
        ]
    )
    return {"risk_aggressive_responses": [str(response.content).strip()]}
