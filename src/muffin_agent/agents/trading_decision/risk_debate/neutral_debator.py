"""Neutral Risk Debator node."""

from __future__ import annotations

import operator
from typing import Annotated, Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from typing_extensions import TypedDict

from ....model_config import ModelConfiguration
from ....prompts import render_template
from .._debate import format_risk_history


class NeutralDebatorInputState(TypedDict, total=False):
    """State keys read by ``neutral_debator_node``."""

    analysis_context: dict[str, Any]
    investment_judge: dict[str, Any]
    trader: dict[str, Any]
    risk_aggressive_responses: Annotated[list[str], operator.add]
    risk_conservative_responses: Annotated[list[str], operator.add]
    risk_neutral_responses: Annotated[list[str], operator.add]


class NeutralDebatorOutputState(TypedDict, total=False):
    """State keys written by ``neutral_debator_node``."""

    risk_neutral_responses: Annotated[list[str], operator.add]


async def neutral_debator_node(
    state: NeutralDebatorInputState, config: RunnableConfig
) -> NeutralDebatorOutputState:
    """One Neutral Risk Debator turn."""
    analysis_context = state["analysis_context"]
    aggressives = state.get("risk_aggressive_responses") or []
    conservatives = state.get("risk_conservative_responses") or []
    neutrals = state.get("risk_neutral_responses") or []

    cfg = ModelConfiguration.from_runnable_config(config)
    primary, *fallbacks = cfg.get_llm_for_role("reasoner")
    llm = (primary.with_fallbacks(fallbacks) if fallbacks else primary).with_retry(
        stop_after_attempt=3, wait_exponential_jitter=True
    )

    prompt = render_template(
        "trading_decision/risk_debate/neutral.jinja",
        ticker=analysis_context.get("ticker", ""),
        query=analysis_context.get("query"),
        investment_judge=state["investment_judge"],
        trader=state["trader"],
        risk_debate_history=format_risk_history(aggressives, conservatives, neutrals),
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
    return {"risk_neutral_responses": [str(response.content).strip()]}
