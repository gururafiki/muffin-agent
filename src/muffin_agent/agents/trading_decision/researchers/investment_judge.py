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

    analysis_context: dict[str, Any]
    investment_bull_responses: Annotated[list[str], operator.add]
    investment_bear_responses: Annotated[list[str], operator.add]


class InvestmentJudgeOutputState(TypedDict, total=False):
    """State keys written by ``investment_judge_node``."""

    investment_judge: dict[str, Any]


async def investment_judge_node(
    state: InvestmentJudgeInputState, config: RunnableConfig
) -> InvestmentJudgeOutputState:
    """Synthesise the completed Bull/Bear debate into ``InvestmentJudgeOutput``."""
    analysis_context = state["analysis_context"]
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
        ticker=analysis_context.get("ticker", ""),
        query=analysis_context.get("query"),
        debate_history=format_debate_history(bulls, bears),
        market_regime=analysis_context.get("market_regime"),
        sector_view=analysis_context.get("sector_view"),
        company_analysis=analysis_context.get("company_analysis"),
        forecast=analysis_context.get("forecast"),
        risk_assessment=analysis_context.get("risk_assessment"),
        valuation=analysis_context.get("valuation"),
        narrative=analysis_context.get("narrative"),
        additional_context=analysis_context.get("additional_context") or {},
    )

    result: InvestmentJudgeOutput = await llm.ainvoke(
        [
            SystemMessage(prompt),
            HumanMessage("Synthesise the debate now."),
        ]
    )
    return {"investment_judge": result.model_dump()}
