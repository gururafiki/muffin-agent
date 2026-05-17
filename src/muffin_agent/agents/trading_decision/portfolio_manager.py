"""Portfolio Manager agent factory.

Final synthesis judge of the trading-decision pipeline. Reads the Judge's
thesis, the Trader's proposal, and the 3-way risk debate transcript, and
produces the canonical ``PortfolioDecisionOutput`` (5-tier rating + final
operational fields + key remaining risks + confidence).

Uses the primary reasoner model — this is the second deep-synthesis point
in the pipeline (the first being the Investment Judge) and the call that
the downstream UI / CLI / reflection memory treats as canonical.
"""

from __future__ import annotations

from langchain.agents.structured_output import AutoStrategy
from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph

from ...model_config import ModelConfiguration
from ...prompts import render_template
from ...utils.agent_builder import MuffinAgentBuilder
from .schemas import PortfolioDecisionOutput


async def create_portfolio_manager_agent(
    config: RunnableConfig,
    *,
    ticker: str,
    query: str | None,
    context_vars: dict,
    investment_judge: dict,
    trader: dict,
    risk_debate_history: str,
) -> CompiledStateGraph:
    """Build the Portfolio Manager.

    Args:
        config: LangGraph ``RunnableConfig``.
        ticker: Equity ticker symbol.
        query: Original investment mandate (or ``None``).
        context_vars: ``AnalysisContext`` fields used as the evidence base.
        investment_judge: ``InvestmentJudgeOutput.model_dump()`` — the
            directional thesis the Trader translated.
        trader: ``TraderOutput.model_dump()`` — the operational proposal
            the risk debate stress-tested.
        risk_debate_history: Full 3-way risk debate transcript.
    """
    configuration = ModelConfiguration.from_runnable_config(config)
    primary, *fallbacks = configuration.get_llm_for_role("reasoner")
    summariser = configuration.get_summariser()

    prompt = render_template(
        "trading_decision/portfolio_manager.jinja",
        ticker=ticker,
        query=query,
        investment_judge=investment_judge,
        trader=trader,
        risk_debate_history=risk_debate_history,
        **context_vars,
    )

    builder = (
        MuffinAgentBuilder(primary, name="portfolio_manager")
        .with_system_prompt(prompt)
        .with_fallback_models(*fallbacks)
        .with_response_format(AutoStrategy(schema=PortfolioDecisionOutput))
    )
    if summariser is not None:
        builder = builder.with_tool_knowledge(summariser)
    return builder.build_react_agent()
