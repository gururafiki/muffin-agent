"""Trader agent factory.

Pure-reasoning ReAct agent with structured output. Translates the
Investment Judge's directional view (signal + conviction + catalysts +
risks) into concrete operational instructions: action, entry/stop/take
profit levels, position sizing, time horizon.

The Trader does not invent new data — it is a translator from a
directional thesis to an executable proposal, anchored in the same
analysis context the Judge used.
"""

from __future__ import annotations

from langchain.agents.structured_output import AutoStrategy
from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph

from ...model_config import ModelConfiguration
from ...prompts import render_template
from ...utils.agent_builder import MuffinAgentBuilder
from .schemas import TraderOutput


async def create_trader_agent(
    config: RunnableConfig,
    *,
    ticker: str,
    query: str | None,
    context_vars: dict,
    investment_judge: dict,
) -> CompiledStateGraph:
    """Build the Trader agent.

    Args:
        config: LangGraph ``RunnableConfig``.
        ticker: Equity ticker symbol.
        query: Original investment mandate (or ``None``).
        context_vars: ``AnalysisContext`` fields used as the evidence base
            (``market_regime``, ``sector_view``, ``valuation``, etc.).
        investment_judge: ``InvestmentJudgeOutput.model_dump()`` — the
            Trader's primary input. Used to derive action, sizing, and
            time horizon.
    """
    configuration = ModelConfiguration.from_runnable_config(config)
    primary, *fallbacks = configuration.get_llm_for_role("reasoner")
    summariser = configuration.get_summariser()

    prompt = render_template(
        "trading_decision/trader.jinja",
        ticker=ticker,
        query=query,
        investment_judge=investment_judge,
        **context_vars,
    )

    builder = (
        MuffinAgentBuilder(primary, name="trader")
        .with_system_prompt(prompt)
        .with_fallback_models(*fallbacks)
        .with_response_format(AutoStrategy(schema=TraderOutput))
    )
    if summariser is not None:
        builder = builder.with_tool_knowledge(summariser)
    return builder.build_react_agent()
