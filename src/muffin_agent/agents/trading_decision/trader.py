"""Trader node — translates the Judge's signal into an operational proposal."""

from __future__ import annotations

from typing import Any, cast

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from typing_extensions import TypedDict

from ...model_config import ModelConfiguration
from ...prompts import render_template
from .schemas import TraderOutput


class TraderInputState(TypedDict, total=False):
    """State keys read by ``trader_node``."""

    ticker: str
    query: str
    narrative: str
    market_report: str
    fundamentals_report: str
    news_report: str
    sentiment_report: str
    investment_judge: dict[str, Any]


class TraderOutputState(TypedDict, total=False):
    """State keys written by ``trader_node``."""

    trader: dict[str, Any]


async def trader_node(
    state: TraderInputState, config: RunnableConfig
) -> TraderOutputState:
    """Translate the Judge's signal into an executable ``TraderOutput``."""
    investment_judge = state["investment_judge"]

    llm = ModelConfiguration.get_chat_model_for_role(
        config, "reasoner", schema=TraderOutput
    )

    prompt = render_template(
        "trading_decision/trader.jinja",
        ticker=state.get("ticker", ""),
        query=state.get("query"),
        narrative=state.get("narrative"),
        market_report=state.get("market_report"),
        fundamentals_report=state.get("fundamentals_report"),
        news_report=state.get("news_report"),
        sentiment_report=state.get("sentiment_report"),
        investment_judge=investment_judge,
    )

    result = cast(
        TraderOutput,
        await llm.ainvoke(
            [
                SystemMessage(prompt),
                HumanMessage("Produce the trade instruction now."),
            ]
        ),
    )
    return {"trader": result.model_dump()}
