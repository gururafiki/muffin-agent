"""News & Macro Analyst — recent news + insider activity via OpenBB."""

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
    "news_company",
    "news_world",
    "equity_ownership_insider_trading",
]


class NewsAnalystOutput(BaseModel):
    """Structured output for the News & Macro Analyst."""

    news_report: str = Field(
        description=(
            "Markdown report covering company-specific news, sector / macro "
            "backdrop, insider activity, and overall sentiment direction. "
            "Ends with a Markdown summary table of the most relevant items."
        )
    )


class NewsAnalystState(AgentState):
    """State schema for the News & Macro Analyst."""

    ticker: Annotated[str, OmitFromSchema(input=False, output=True)]
    decision_date: Annotated[str, OmitFromSchema(input=False, output=True)]
    news_report: Annotated[str, OmitFromSchema(input=True, output=False)]


async def build_news_analyst_agent(
    config: RunnableConfig,
) -> CompiledStateGraph:
    """Build the News & Macro Analyst as a compiled ReAct agent."""
    cfg = ModelConfiguration.from_runnable_config(config)
    primary, *fallbacks = cfg.get_llm_for_role("collector")
    mcp_tools = await get_tools(config, _MCP_TOOLS)

    builder = (
        MuffinAgentBuilder(primary, name="news_analyst")
        .with_fallback_models(*fallbacks)
        .with_state_schema(NewsAnalystState)
        .with_input_prompt_template(
            "trading_decision/analysts/news.jinja",
        )
        .with_response_format(NewsAnalystOutput)
    )
    for tool in mcp_tools:
        builder = builder.with_tool(tool, run_limit=3)
    return builder.build_react_agent()
