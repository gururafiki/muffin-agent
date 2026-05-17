"""Bull Researcher agent factory.

Pure-reasoning ReAct agent (no tools, no subagents). Adversarial debater
that argues for taking or growing the position. Reads an
``AnalysisContext`` plus the running debate history and produces a single
plain-prose argument. The node wrapper in ``nodes.py`` prepends the
speaker tag and updates the debate state.

Standalone invocation:

    from langchain_core.messages import HumanMessage
    from muffin_agent.agents.trading_decision import (
        AnalysisContext,
        create_bull_researcher_agent,
    )

    context = AnalysisContext.from_narrative("AAPL", "...")
    agent = await create_bull_researcher_agent(config)
    payload = {
        "ticker": context.ticker,
        "query": context.query,
        **context.model_dump(exclude={"ticker", "query"}),
        "debate_history": "",
        "opposing_last": "",
    }
    result = await agent.ainvoke({"messages": [HumanMessage("...")]})
"""

from __future__ import annotations

from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph

from ....model_config import ModelConfiguration
from ....prompts import render_template
from ....utils.agent_builder import MuffinAgentBuilder


async def create_bull_researcher_agent(
    config: RunnableConfig,
    *,
    ticker: str,
    query: str | None,
    context_vars: dict,
    debate_history: str,
    opposing_last: str,
) -> CompiledStateGraph:
    """Build the Bull Researcher debater.

    The system prompt is rendered up-front from the supplied per-turn
    variables so the agent only needs a trivial ``HumanMessage`` trigger.
    This keeps prompt-construction logic close to the agent factory and
    lets the node wrapper stay generic.

    Args:
        config: LangGraph ``RunnableConfig``.
        ticker: Equity ticker symbol.
        query: Original investment mandate or analysis focus (or ``None``).
        context_vars: Extra ``AnalysisContext`` fields rendered as evidence
            in the prompt (e.g. ``market_regime``, ``valuation``,
            ``narrative``). All optional — the template uses ``{% if %}``
            guards so missing fields render as nothing.
        debate_history: Full interleaved debate transcript so far.
            Empty string on the opening turn.
        opposing_last: The Bear Researcher's most recent argument.
            Empty string on the opening turn.
    """
    configuration = ModelConfiguration.from_runnable_config(config)
    primary, *fallbacks = configuration.get_llm_for_role("reasoner")
    summariser = configuration.get_summariser()

    prompt = render_template(
        "trading_decision/researchers/bull.jinja",
        ticker=ticker,
        query=query,
        debate_history=debate_history,
        opposing_last=opposing_last,
        **context_vars,
    )

    builder = (
        MuffinAgentBuilder(primary, name="bull_researcher")
        .with_system_prompt(prompt)
        .with_fallback_models(*fallbacks)
    )
    if summariser is not None:
        builder = builder.with_tool_knowledge(summariser)
    return builder.build_react_agent()
