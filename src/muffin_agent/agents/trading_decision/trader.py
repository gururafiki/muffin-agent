"""Trader node — translates the Judge's signal into an operational proposal."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from typing_extensions import TypedDict

from ...model_config import ModelConfiguration
from ...prompts import render_template
from .schemas import TraderOutput


class TraderInputState(TypedDict, total=False):
    """State keys read by ``trader_node``."""

    analysis_context: dict[str, Any]
    investment_judge: dict[str, Any]


class TraderOutputState(TypedDict, total=False):
    """State keys written by ``trader_node``."""

    trader: dict[str, Any]


async def trader_node(
    state: TraderInputState, config: RunnableConfig
) -> TraderOutputState:
    """Translate the Judge's signal into an executable ``TraderOutput``."""
    analysis_context = state["analysis_context"]
    investment_judge = state["investment_judge"]

    cfg = ModelConfiguration.from_runnable_config(config)
    primary, *fallbacks = cfg.get_llm_for_role("reasoner")
    llm = (primary.with_fallbacks(fallbacks) if fallbacks else primary).with_retry(
        stop_after_attempt=3, wait_exponential_jitter=True
    )
    llm = llm.with_structured_output(TraderOutput)

    prompt = render_template(
        "trading_decision/trader.jinja",
        ticker=analysis_context.get("ticker", ""),
        query=analysis_context.get("query"),
        investment_judge=investment_judge,
        market_regime=analysis_context.get("market_regime"),
        sector_view=analysis_context.get("sector_view"),
        company_analysis=analysis_context.get("company_analysis"),
        forecast=analysis_context.get("forecast"),
        risk_assessment=analysis_context.get("risk_assessment"),
        valuation=analysis_context.get("valuation"),
        narrative=analysis_context.get("narrative"),
        additional_context=analysis_context.get("additional_context") or {},
    )

    result: TraderOutput = await llm.ainvoke(
        [
            SystemMessage(prompt),
            HumanMessage("Produce the trade instruction now."),
        ]
    )
    return {"trader": result.model_dump()}
