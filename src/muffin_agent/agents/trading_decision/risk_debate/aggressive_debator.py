"""Aggressive Risk Debator agent factory.

Pure-reasoning ReAct agent (no tools, no subagents). One of three risk
debaters that stress-test the Trader's proposal from a sharply-defined
persona perspective. Champions higher conviction / larger sizing / more
aggressive entry when the evidence supports it.
"""

from __future__ import annotations

from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph

from ....model_config import ModelConfiguration
from ....prompts import render_template
from ....utils.agent_builder import MuffinAgentBuilder


async def create_aggressive_debator_agent(
    config: RunnableConfig,
    *,
    ticker: str,
    query: str | None,
    context_vars: dict,
    investment_judge: dict,
    trader: dict,
    debate_history: str,
    opposing_last: str,
) -> CompiledStateGraph:
    """Build the Aggressive Risk Debator.

    Args:
        config: LangGraph ``RunnableConfig``.
        ticker: Equity ticker symbol.
        query: Original investment mandate (or ``None``).
        context_vars: ``AnalysisContext`` fields (market_regime, sector_view,
            company_analysis, forecast, risk_assessment, valuation,
            narrative, additional_context).
        investment_judge: ``InvestmentJudgeOutput.model_dump()``.
        trader: ``TraderOutput.model_dump()`` — the proposal being
            stress-tested.
        debate_history: Full risk debate transcript so far.
        opposing_last: The most recent Conservative or Neutral argument
            (empty string on the opening turn).
    """
    configuration = ModelConfiguration.from_runnable_config(config)
    primary, *fallbacks = configuration.get_llm_for_role("reasoner")
    summariser = configuration.get_summariser()

    prompt = render_template(
        "trading_decision/risk_debate/aggressive.jinja",
        ticker=ticker,
        query=query,
        investment_judge=investment_judge,
        trader=trader,
        debate_history=debate_history,
        opposing_last=opposing_last,
        **context_vars,
    )

    builder = (
        MuffinAgentBuilder(primary, name="aggressive_debator")
        .with_system_prompt(prompt)
        .with_fallback_models(*fallbacks)
    )
    if summariser is not None:
        builder = builder.with_tool_knowledge(summariser)
    return builder.build_react_agent()
