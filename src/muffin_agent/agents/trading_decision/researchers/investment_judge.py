"""Investment Judge node — single LLM call with structured output."""

from __future__ import annotations

import operator
from typing import Annotated, Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from typing_extensions import TypedDict

from ....model_config import ModelConfiguration
from ....prompts import render_template
from .._debate import format_debate_history
from ..schemas import InvestmentJudgeOutput


class InvestmentJudgeInputState(TypedDict, total=False):
    """State keys read by ``investment_judge_node``."""

    ticker: str
    query: str
    narrative: str
    market_report: str
    fundamentals_report: str
    news_report: str
    sentiment_report: str
    investment_bull_responses: Annotated[list[str], operator.add]
    investment_bear_responses: Annotated[list[str], operator.add]


class InvestmentJudgeOutputState(TypedDict, total=False):
    """State keys written by ``investment_judge_node``."""

    investment_judge: dict[str, Any]


async def investment_judge_node(
    state: InvestmentJudgeInputState, config: RunnableConfig
) -> InvestmentJudgeOutputState:
    """Synthesise the completed Bull/Bear debate into ``InvestmentJudgeOutput``."""
    bulls = state.get("investment_bull_responses") or []
    bears = state.get("investment_bear_responses") or []

    cfg = ModelConfiguration.from_runnable_config(config)
    primary, *fallbacks = cfg.get_llm_for_role("reasoner")
    llm = (primary.with_fallbacks(fallbacks) if fallbacks else primary).with_retry(
        stop_after_attempt=3, wait_exponential_jitter=True
    )
    llm = llm.with_structured_output(InvestmentJudgeOutput)

    prompt = render_template(
        "trading_decision/researchers/investment_judge.jinja",
        ticker=state.get("ticker", ""),
        query=state.get("query"),
        narrative=state.get("narrative"),
        market_report=state.get("market_report"),
        fundamentals_report=state.get("fundamentals_report"),
        news_report=state.get("news_report"),
        sentiment_report=state.get("sentiment_report"),
        debate_history=format_debate_history(bulls, bears),
    )

    result: InvestmentJudgeOutput = await llm.ainvoke(
        [
            SystemMessage(prompt),
            HumanMessage("Synthesise the debate now."),
        ]
    )
    return {"investment_judge": result.model_dump()}
