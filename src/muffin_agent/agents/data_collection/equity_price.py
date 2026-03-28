"""Equity Price data collection agent.

ReAct agent that retrieves stock price data (quotes, historical OHLCV,
performance, market cap, spreads) via OpenBB MCP tools.
"""

from langchain.agents import create_agent
from langchain_core.runnables import RunnableConfig

from ...model_config import ModelConfiguration
from ...prompts import render_template
from ...sandbox.tools import execute_python
from .utils import data_collection_middleware, get_tools

MCP_TOOLS = [
    "equity_historical_market_cap",
    "equity_price_historical",
    "equity_price_nbbo",
    "equity_price_performance",
    "equity_price_quote",
]


async def create_equity_price_data_collection_agent(config: RunnableConfig):
    """Build the equity price ReAct agent."""
    tools = await get_tools(config, MCP_TOOLS, custom_tools=[execute_python])
    prompt = render_template("data_collection/equity_price.jinja")
    model_config = ModelConfiguration.from_runnable_config(config)
    llm = model_config.get_llm()
    return create_agent(
        model=llm,
        tools=tools,
        system_prompt=prompt,
        middleware=data_collection_middleware(MCP_TOOLS),
    )
