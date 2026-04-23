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
    llm = ModelConfiguration.from_runnable_config(config).get_llm()

    builder = (
        MuffinAgentBuilder(llm, name="fama_french")
        .with_system_prompt_template("data_collection/fama_french.jinja")
        .with_short_term_memory()
    )
    for tool in tools:
        builder = builder.with_tool(tool)
    return builder.build_react_agent()
