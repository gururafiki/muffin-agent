"""Equity Fundamentals data collection agent.

ReAct agent that retrieves company fundamental data (financial statements,
ratios, metrics, etc.) via OpenBB MCP tools.
"""

from langchain.agents import create_agent

from ...config import Configuration
from ...prompts import render_template
from .utils import ToolErrorHandler, get_tools

MCP_TOOLS = [
    "equity_fundamental_balance",
    "equity_fundamental_balance_growth",
    "equity_fundamental_cash",
    "equity_fundamental_cash_growth",
    "equity_fundamental_dividends",
    "equity_fundamental_employee_count",
    "equity_fundamental_esg_score",
    "equity_fundamental_filings",
    "equity_fundamental_historical_attributes",
    "equity_fundamental_historical_eps",
    "equity_fundamental_historical_splits",
    "equity_fundamental_income",
    "equity_fundamental_income_growth",
    "equity_fundamental_latest_attributes",
    "equity_fundamental_management",
    "equity_fundamental_management_compensation",
    "equity_fundamental_management_discussion_analysis",
    "equity_fundamental_metrics",
    "equity_fundamental_ratios",
    "equity_fundamental_reported_financials",
    "equity_fundamental_revenue_per_geography",
    "equity_fundamental_revenue_per_segment",
    "equity_fundamental_search_attributes",
    "equity_fundamental_trailing_dividend_yield",
    "equity_fundamental_transcript",
]


async def create_equity_fundamentals_data_collection_agent(config: Configuration):
    """Build the equity fundamentals ReAct agent."""
    tools = await get_tools(config, MCP_TOOLS)
    prompt = render_template("equity_fundamentals.jinja")
    llm = config.get_llm()
    return create_agent(
        model=llm,
        tools=tools,
        system_prompt=prompt,
        middleware=[ToolErrorHandler()],
    )
