"""Neutral Risk Debator agent factory.

Pure-reasoning ReAct agent (no tools, no subagents). Third leg of the
risk debate trio: argues for a balanced scaled position that captures
the upside Aggressive is pressing while honouring the downside
Conservative is flagging. Calls out where either side over-presses.
"""

from __future__ import annotations

from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph

from ....model_config import ModelConfiguration
from ....prompts import render_template
from ....utils.agent_builder import MuffinAgentBuilder


async def create_neutral_debator_agent(
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
    """Build the Neutral Risk Debator. See aggressive counterpart for docs."""
    configuration = ModelConfiguration.from_runnable_config(config)
    primary, *fallbacks = configuration.get_llm_for_role("reasoner")
    summariser = configuration.get_summariser()

    prompt = render_template(
        "trading_decision/risk_debate/neutral.jinja",
        ticker=ticker,
        query=query,
        investment_judge=investment_judge,
        trader=trader,
        debate_history=debate_history,
        opposing_last=opposing_last,
        **context_vars,
    )

    builder = (
        MuffinAgentBuilder(primary, name="neutral_debator")
        .with_system_prompt(prompt)
        .with_fallback_models(*fallbacks)
    )
    if summariser is not None:
        builder = builder.with_tool_knowledge(summariser)
    return builder.build_react_agent()
