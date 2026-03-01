"""Equity Estimates data collection agent.

ReAct agent that retrieves analyst estimates data (price targets, consensus
estimates, forward EPS/EBITDA/PE/sales) via OpenBB MCP tools.
"""

from langchain.agents import create_agent

from ...config import Configuration
from ...prompts import render_template
from .utils import ToolErrorHandler, get_tools

MCP_TOOLS = [
    "equity_estimates_consensus",
    "equity_estimates_forward_ebitda",
    "equity_estimates_forward_eps",
    "equity_estimates_forward_pe",
    "equity_estimates_forward_sales",
    "equity_estimates_historical",
    "equity_estimates_price_target",
    "equity_estimates_price_target_consensus",
]


async def create_equity_estimates_data_collection_agent(config: Configuration):
    """Build the equity estimates ReAct agent."""
    tools = await get_tools(config, MCP_TOOLS)
    prompt = render_template("equity_estimates.jinja")
    llm = config.get_llm()
    return create_agent(
        model=llm,
        tools=tools,
        system_prompt=prompt,
        middleware=[ToolErrorHandler()],
    )
