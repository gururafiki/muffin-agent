"""Fundamentals Analyst — financial-statement analysis via OpenBB."""

from __future__ import annotations

from typing import Annotated

from langchain.agents import AgentState
from langchain.agents.middleware.types import OmitFromSchema
from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field

from ....model_config import ModelConfiguration
from ....utils.agent_builder import MuffinAgentBuilder
from ...data_collection.utils import get_tools

_MCP_TOOLS = [
    "equity_fundamental_balance",
    "equity_fundamental_income",
    "equity_fundamental_cash",
    "equity_fundamental_ratios",
    "equity_fundamental_metrics",
    "equity_fundamental_historical_eps",
    "equity_fundamental_dividends",
]


class FundamentalsAnalystOutput(BaseModel):
    """Structured output for the Fundamentals Analyst."""

    fundamentals_report: str = Field(
        description=(
            "Markdown report covering income statement, balance sheet, "
            "cash flow, returns, balance-sheet health, capital allocation, "
            "and quality flags. Ends with a Markdown summary table of "
            "headline metrics."
        )
    )


class FundamentalsAnalystState(AgentState):
    """State schema for the Fundamentals Analyst."""

    ticker: Annotated[str, OmitFromSchema(input=False, output=True)]
    decision_date: Annotated[str, OmitFromSchema(input=False, output=True)]
    fundamentals_report: Annotated[str, OmitFromSchema(input=True, output=False)]


async def build_fundamentals_analyst_agent(
    config: RunnableConfig,
) -> CompiledStateGraph:
    """Build the Fundamentals Analyst as a compiled ReAct agent."""
    cfg = ModelConfiguration.from_runnable_config(config)
    primary, *fallbacks = cfg.get_llm_for_role("collector")
    mcp_tools = await get_tools(config, _MCP_TOOLS)

    builder = (
        MuffinAgentBuilder(primary, name="fundamentals_analyst")
        .with_fallback_models(*fallbacks)
        .with_state_schema(FundamentalsAnalystState)
        .with_runtime_system_prompt_template(
            "trading_decision/analysts/fundamentals.jinja",
        )
        .with_response_format(FundamentalsAnalystOutput)
    )
    for tool in mcp_tools:
        builder = builder.with_tool(tool, run_limit=3)
    return builder.build_react_agent()
