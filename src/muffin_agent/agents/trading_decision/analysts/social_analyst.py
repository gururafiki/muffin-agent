"""Social Sentiment Analyst — retail / social-media reads via news + firecrawl."""

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
    "firecrawl_search",
]


class SocialAnalystOutput(BaseModel):
    """Structured output for the Social Sentiment Analyst."""

    sentiment_report: str = Field(
        description=(
            "Long-form Markdown report covering sentiment baseline, day-by-day "
            "drift, source breakdown (Reddit / X / news / podcasts), "
            "retail-vs-institutional read, and implications for traders. "
            "Ends with a Markdown summary table of key sources + signals."
        )
    )


class SocialAnalystState(AgentState):
    """State schema for the Social Sentiment Analyst."""

    ticker: Annotated[str, OmitFromSchema(input=False, output=True)]
    decision_date: Annotated[str, OmitFromSchema(input=False, output=True)]
    sentiment_report: Annotated[str, OmitFromSchema(input=True, output=False)]


async def build_social_analyst_agent(
    config: RunnableConfig,
) -> CompiledStateGraph:
    """Build the Social Sentiment Analyst as a compiled ReAct agent."""
    cfg = ModelConfiguration.from_runnable_config(config)
    primary, *fallbacks = cfg.get_llm_for_role("collector")
    mcp_tools = await get_tools(config, _MCP_TOOLS)

    builder = (
        MuffinAgentBuilder(primary, name="social_analyst")
        .with_fallback_models(*fallbacks)
        .with_state_schema(SocialAnalystState)
        .with_input_prompt_template(
            "trading_decision/analysts/social.jinja",
        )
        .with_response_format(SocialAnalystOutput)
    )
    for tool in mcp_tools:
        builder = builder.with_tool(tool, run_limit=4)
    return builder.build_react_agent()
