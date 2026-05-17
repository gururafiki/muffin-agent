"""Investment Judge agent factory.

Pure-reasoning ReAct agent with structured output. Synthesises a completed
Bull vs Bear debate into an ``InvestmentJudgeOutput`` (signal, conviction,
synthesised bull/bear cases, catalysts, risks, monitoring checklist,
winning side, reasoning).

Uses the primary reasoner model — synthesis is the expensive step and gets
the strongest available model in the role chain.
"""

from __future__ import annotations

from langchain.agents.structured_output import AutoStrategy
from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph

from ....model_config import ModelConfiguration
from ....prompts import render_template
from ....utils.agent_builder import MuffinAgentBuilder
from ..schemas import InvestmentJudgeOutput


async def create_investment_judge_agent(
    config: RunnableConfig,
    *,
    ticker: str,
    query: str | None,
    context_vars: dict,
    debate_history: str,
) -> CompiledStateGraph:
    """Build the Investment Judge synthesis agent.

    Args:
        config: LangGraph ``RunnableConfig``.
        ticker: Equity ticker symbol.
        query: Original investment mandate or analysis focus (or ``None``).
        context_vars: ``AnalysisContext`` fields rendered as evidence in
            the prompt.
        debate_history: Full debate transcript to synthesise.
    """
    configuration = ModelConfiguration.from_runnable_config(config)
    primary, *fallbacks = configuration.get_llm_for_role("reasoner")
    summariser = configuration.get_summariser()

    prompt = render_template(
        "trading_decision/researchers/investment_judge.jinja",
        ticker=ticker,
        query=query,
        debate_history=debate_history,
        **context_vars,
    )

    builder = (
        MuffinAgentBuilder(primary, name="investment_judge")
        .with_system_prompt(prompt)
        .with_fallback_models(*fallbacks)
        .with_response_format(AutoStrategy(schema=InvestmentJudgeOutput))
    )
    if summariser is not None:
        builder = builder.with_tool_knowledge(summariser)
    return builder.build_react_agent()
