"""ETF and index data collection agent.

ReAct agent that retrieves ETF info, sector weights, holdings, index levels,
and S&P 500 valuation multiples via OpenBB MCP tools.
"""

from langchain.agents import create_agent

from ...config import Configuration
from ...prompts import render_template
from .utils import ToolErrorHandler, get_tools

MCP_TOOLS = [
    "etf_countries",
    "etf_discovery_active",
    "etf_discovery_gainers",
    "etf_discovery_losers",
    "etf_equity_exposure",
    "etf_historical",
    "etf_holdings",
    "etf_info",
    "etf_nport_disclosure",
    "etf_price_performance",
    "etf_search",
    "etf_sectors",
    "index_available",
    "index_constituents",
    "index_price_historical",
    "index_search",
    "index_sectors",
    "index_snapshots",
    "index_sp500_multiples",
]


async def create_etf_index_data_collection_agent(config: Configuration):
    """Build the ETF and index data ReAct agent."""
    tools = await get_tools(config, MCP_TOOLS)
    prompt = render_template("etf_index.jinja")
    llm = config.get_llm()
    return create_agent(
        model=llm,
        tools=tools,
        system_prompt=prompt,
        middleware=[ToolErrorHandler()],
    )
