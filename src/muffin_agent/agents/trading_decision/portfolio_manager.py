"""Portfolio Manager node — canonical final synthesis."""

from __future__ import annotations

from typing import Annotated, Any, cast

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from ...model_config import ModelConfiguration
from ...prompts import render_template
from ._debate import format_risk_history
from .schemas import PortfolioDecisionOutput


class PortfolioManagerInputState(TypedDict, total=False):
    """State keys read by ``portfolio_manager_node``."""

    ticker: str
    query: str
    narrative: str
    market_report: str
    fundamentals_report: str
    news_report: str
    sentiment_report: str
    investment_judge: dict[str, Any]
    trader: dict[str, Any]
    risk_debate_messages: Annotated[list[BaseMessage], add_messages]
    past_reflections: str


class PortfolioManagerOutputState(TypedDict, total=False):
    """State keys written by ``portfolio_manager_node``."""

    portfolio_decision: dict[str, Any]


async def portfolio_manager_node(
    state: PortfolioManagerInputState, config: RunnableConfig
) -> PortfolioManagerOutputState:
    """Synthesise Judge + Trader + risk debate into ``PortfolioDecisionOutput``.

    Also pulls in ``past_reflections`` when present so the PM prompt can
    reference past lessons from the reflection log.
    """
    risk_messages = state.get("risk_debate_messages") or []

    llm = ModelConfiguration.get_chat_model_for_role(
        config, "reasoner", schema=PortfolioDecisionOutput
    )

    prompt = render_template(
        "trading_decision/portfolio_manager.jinja",
        ticker=state.get("ticker", ""),
        query=state.get("query"),
        narrative=state.get("narrative"),
        market_report=state.get("market_report"),
        fundamentals_report=state.get("fundamentals_report"),
        news_report=state.get("news_report"),
        sentiment_report=state.get("sentiment_report"),
        investment_judge=state["investment_judge"],
        trader=state["trader"],
        transcript=format_risk_history(risk_messages),
        past_reflections=state.get("past_reflections") or "",
    )

    result = cast(
        PortfolioDecisionOutput,
        await llm.ainvoke(
            [
                SystemMessage(prompt),
                HumanMessage("Produce the portfolio decision now."),
            ]
        ),
    )
    return {"portfolio_decision": result.model_dump()}
