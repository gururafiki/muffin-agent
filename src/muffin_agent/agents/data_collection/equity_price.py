"""Equity Price data collection agent.

ReAct agent that retrieves stock price data (quotes, historical OHLCV,
performance, market cap, spreads) via OpenBB MCP tools.
"""

from langchain.agents import create_agent

from ...config import Configuration
from ...prompts import render_template
from ...sandbox import create_opensandbox_backend, create_python_execution_tool
from .utils import ToolErrorHandler, get_tools

MCP_TOOLS = [
    "equity_historical_market_cap",
    "equity_price_historical",
    "equity_price_nbbo",
    "equity_price_performance",
    "equity_price_quote",
]


async def create_equity_price_data_collection_agent(config: Configuration):
    """Build the equity price ReAct agent."""
    backend = await create_opensandbox_backend(config)
    python_tool = create_python_execution_tool(backend)
    tools = await get_tools(config, MCP_TOOLS, custom_tools=[python_tool])
    prompt = render_template("equity_price.jinja")
    llm = config.get_llm()
    return create_agent(
        model=llm,
        tools=tools,
        system_prompt=prompt,
        middleware=[ToolErrorHandler()],
    )
