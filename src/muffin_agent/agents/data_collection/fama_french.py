"""Fama-French factor data collection agent.

ReAct agent that retrieves Fama-French 3/5-factor model returns, portfolio
returns, and size/value breakpoints via OpenBB MCP tools.
"""

from langchain.agents import create_agent

from ...config import Configuration
from ...prompts import render_template
from .utils import ToolErrorHandler, get_tools

MCP_TOOLS = [
    "famafrench_breakpoints",
    "famafrench_country_portfolio_returns",
    "famafrench_factors",
    "famafrench_international_index_returns",
    "famafrench_regional_portfolio_returns",
    "famafrench_us_portfolio_returns",
]


async def create_fama_french_data_collection_agent(config: Configuration):
    """Build the Fama-French factor data ReAct agent."""
    tools = await get_tools(config, MCP_TOOLS)
    prompt = render_template("fama_french.jinja")
    llm = config.get_llm()
    return create_agent(
        model=llm,
        tools=tools,
        system_prompt=prompt,
        middleware=[ToolErrorHandler()],
    )
