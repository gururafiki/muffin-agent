"""Equity Fundamentals data collection agent.

ReAct agent that retrieves company fundamental data (financial statements,
ratios, metrics, etc.) via OpenBB MCP tools.
"""

from langchain.agents import create_agent
from langchain.agents.middleware import wrap_tool_call
from langchain_core.messages import ToolMessage
from langchain_mcp_adapters.client import MultiServerMCPClient

from muffin_agent.config import Configuration
from muffin_agent.prompts import render_template

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


async def get_tools(config: Configuration) -> list:
    """Load MCP tools filtered to fundamentals, plus any custom tools."""
    client = MultiServerMCPClient(config.get_mcp_connections())
    all_tools = await client.get_tools()
    mcp_tools = [t for t in all_tools if t.name in MCP_TOOLS]
    custom_tools: list = []
    return mcp_tools + custom_tools


@wrap_tool_call
async def handle_tool_errors(request, handler):
    """Catch tool exceptions and return error messages to the agent."""
    try:
        return await handler(request)
    except Exception as e:
        return ToolMessage(
            content=f"Error: {e!s}",
            tool_call_id=request.tool_call["id"],
        )


async def build_graph(config: Configuration):
    """Build the equity fundamentals ReAct agent."""
    tools = await get_tools(config)
    prompt = render_template("equity_fundamentals.jinja")
    llm = config.get_llm()
    return create_agent(
        model=llm,
        tools=tools,
        system_prompt=prompt,
        middleware=[handle_tool_errors],
    )
