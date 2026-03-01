"""Equity Ownership & Short Interest data collection agent.

ReAct agent that retrieves ownership structure and short interest data
(major holders, institutional ownership, insider trades, 13F filings,
short interest/volume/FTDs) via OpenBB MCP tools.
"""

from langchain.agents import create_agent

from ...config import Configuration
from ...prompts import render_template
from .utils import ToolErrorHandler, get_tools

MCP_TOOLS = [
    "equity_ownership_form_13f",
    "equity_ownership_government_trades",
    "equity_ownership_insider_trading",
    "equity_ownership_institutional",
    "equity_ownership_major_holders",
    "equity_ownership_share_statistics",
    "equity_shorts_fails_to_deliver",
    "equity_shorts_short_interest",
    "equity_shorts_short_volume",
]


async def create_equity_ownership_data_collection_agent(config: Configuration):
    """Build the equity ownership & short interest ReAct agent."""
    tools = await get_tools(config, MCP_TOOLS)
    prompt = render_template("equity_ownership.jinja")
    llm = config.get_llm()
    return create_agent(
        model=llm,
        tools=tools,
        system_prompt=prompt,
        middleware=[ToolErrorHandler()],
    )
