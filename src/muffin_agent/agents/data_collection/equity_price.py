"""Equity Price data collection agent.

ReAct agent that retrieves stock price data (quotes, historical OHLCV,
performance, market cap, spreads) via OpenBB MCP tools.
"""

from langchain.agents import create_agent

from ...config import Configuration
from ...prompts import render_template
from .utils import get_tools, handle_tool_errors

MCP_TOOLS = [
    "equity_historical_market_cap",
    "equity_price_historical",
    "equity_price_nbbo",
    "equity_price_performance",
    "equity_price_quote",
]


async def build_graph(config: Configuration):
    """Build the equity price ReAct agent."""
    tools = await get_tools(config, MCP_TOOLS)
    prompt = render_template("equity_price.jinja")
    llm = config.get_llm()
    return create_agent(
        model=llm,
        tools=tools,
        system_prompt=prompt,
        middleware=[handle_tool_errors],
    )
