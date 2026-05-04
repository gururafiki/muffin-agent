"""Fama-French factor data collection agent.

ReAct agent that retrieves Fama-French 3/5-factor model returns, portfolio
returns, and size/value breakpoints via OpenBB MCP tools.
"""

from langchain_core.runnables import RunnableConfig

from ...model_config import ModelConfiguration
from ...utils.agent_builder import MuffinAgentBuilder
from .utils import get_tools

MCP_TOOLS = [
    "famafrench_breakpoints",
    "famafrench_country_portfolio_returns",
    "famafrench_factors",
    "famafrench_international_index_returns",
    "famafrench_regional_portfolio_returns",
    "famafrench_us_portfolio_returns",
]


async def create_fama_french_data_collection_agent(config: RunnableConfig):
    """Build the Fama-French factor data ReAct agent."""
    tools = await get_tools(config, MCP_TOOLS)
    configuration = ModelConfiguration.from_runnable_config(config)
    primary, *fallbacks = configuration.get_llm_for_role("collector")
    summariser = configuration.get_summariser()

    builder = (
        MuffinAgentBuilder(primary, name="fama_french")
        .with_system_prompt_template("data_collection/fama_french.jinja")
        .with_fallback_models(*fallbacks)
        .with_short_term_memory()
    )
    if summariser is not None:
        builder = builder.with_tool_knowledge(summariser)
    for tool in tools:
        builder = builder.with_tool(tool)
    return builder.build_react_agent()
