"""Conservative Risk Debator agent factory.

Pure-reasoning ReAct agent (no tools, no subagents). Symmetric counterpart
to ``aggressive_debator``: argues for tighter risk control, smaller
sizing, wider stops, or shorter horizons when the evidence supports
caution.
"""

from __future__ import annotations

from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph

from ....model_config import ModelConfiguration
from ....prompts import render_template
from ....utils.agent_builder import MuffinAgentBuilder


async def create_conservative_debator_agent(
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
    """Build the Conservative Risk Debator. See aggressive counterpart for docs."""
    configuration = ModelConfiguration.from_runnable_config(config)
    primary, *fallbacks = configuration.get_llm_for_role("reasoner")
    summariser = configuration.get_summariser()

    prompt = render_template(
        "trading_decision/risk_debate/conservative.jinja",
        ticker=ticker,
        query=query,
        investment_judge=investment_judge,
        trader=trader,
        debate_history=debate_history,
        opposing_last=opposing_last,
        **context_vars,
    )

    builder = (
        MuffinAgentBuilder(primary, name="conservative_debator")
        .with_system_prompt(prompt)
        .with_fallback_models(*fallbacks)
    )
    if summariser is not None:
        builder = builder.with_tool_knowledge(summariser)
    return builder.build_react_agent()
