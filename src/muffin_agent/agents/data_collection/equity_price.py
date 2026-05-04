"""Equity Price data collection agent.

ReAct agent that retrieves stock price data (quotes, historical OHLCV,
performance, market cap, spreads) via OpenBB MCP tools.
"""

from langchain_core.runnables import RunnableConfig

from ...model_config import ModelConfiguration
from ...sandbox.tools import execute_python
from ...utils.agent_builder import MuffinAgentBuilder
from .utils import get_tools

MCP_TOOLS = [
    "equity_historical_market_cap",
    "equity_price_historical",
    "equity_price_nbbo",
    "equity_price_performance",
    "equity_price_quote",
]


async def create_equity_price_data_collection_agent(config: RunnableConfig):
    """Build the equity price ReAct agent."""
    tools = await get_tools(config, MCP_TOOLS)
    configuration = ModelConfiguration.from_runnable_config(config)
    primary, *fallbacks = configuration.get_llm_for_role("collector")
    summariser = configuration.get_summariser()

    builder = (
        MuffinAgentBuilder(primary, name="equity_price")
        .with_system_prompt_template("data_collection/equity_price.jinja")
        .with_fallback_models(*fallbacks)
        .with_sandbox()
        .with_short_term_memory()
    )
    if summariser is not None:
        builder = builder.with_tool_knowledge(summariser)
    for tool in tools:
        builder = builder.with_tool(tool)
    # execute_python is stateful / not cacheable.
    builder = builder.with_tool(execute_python, is_cacheable=False)
    return builder.build_react_agent()
