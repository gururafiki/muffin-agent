"""Bear Researcher agent factory.

Pure-reasoning ReAct agent (no tools, no subagents). Adversarial debater
that argues against taking or growing the position. Symmetric counterpart
to ``bull_researcher``.
"""

from __future__ import annotations

from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph

from ....model_config import ModelConfiguration
from ....prompts import render_template
from ....utils.agent_builder import MuffinAgentBuilder


async def create_bear_researcher_agent(
    config: RunnableConfig,
    *,
    ticker: str,
    query: str | None,
    context_vars: dict,
    debate_history: str,
    opposing_last: str,
) -> CompiledStateGraph:
    """Build the Bear Researcher debater.

    See ``create_bull_researcher_agent`` for full parameter docs — this
    is the symmetric counterpart that argues the bearish side.
    """
    configuration = ModelConfiguration.from_runnable_config(config)
    primary, *fallbacks = configuration.get_llm_for_role("reasoner")
    summariser = configuration.get_summariser()

    prompt = render_template(
        "trading_decision/researchers/bear.jinja",
        ticker=ticker,
        query=query,
        debate_history=debate_history,
        opposing_last=opposing_last,
        **context_vars,
    )

    builder = (
        MuffinAgentBuilder(primary, name="bear_researcher")
        .with_system_prompt(prompt)
        .with_fallback_models(*fallbacks)
    )
    if summariser is not None:
        builder = builder.with_tool_knowledge(summariser)
    return builder.build_react_agent()
