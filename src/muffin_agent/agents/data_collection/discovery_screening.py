"""Discovery and screening data collection agent.

ReAct agent that retrieves market-wide equity screener results, discovery
screens, calendars, peer comparisons, and company profiles via OpenBB MCP tools.
"""

from langchain.agents import create_agent

from ...config import Configuration
from ...prompts import render_template
from .utils import ToolErrorHandler, get_tools

MCP_TOOLS = [
    "equity_calendar_dividend",
    "equity_calendar_earnings",
    "equity_calendar_events",
    "equity_calendar_ipo",
    "equity_calendar_splits",
    "equity_compare_company_facts",
    "equity_compare_groups",
    "equity_compare_peers",
    "equity_darkpool_otc",
    "equity_discovery_active",
    "equity_discovery_aggressive_small_caps",
    "equity_discovery_filings",
    "equity_discovery_gainers",
    "equity_discovery_growth_tech",
    "equity_discovery_latest_financial_reports",
    "equity_discovery_losers",
    "equity_discovery_top_retail",
    "equity_discovery_undervalued_growth",
    "equity_discovery_undervalued_large_caps",
    "equity_market_snapshots",
    "equity_profile",
    "equity_screener",
    "equity_search",
]


async def create_discovery_screening_data_collection_agent(config: Configuration):
    """Build the discovery and screening data ReAct agent."""
    tools = await get_tools(config, MCP_TOOLS)
    prompt = render_template("discovery_screening.jinja")
    llm = config.get_llm()
    return create_agent(
        model=llm,
        tools=tools,
        system_prompt=prompt,
        middleware=[ToolErrorHandler()],
    )
