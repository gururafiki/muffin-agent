"""Equity Estimates data collection agent.

ReAct agent that retrieves analyst estimates data (price targets, consensus
estimates, forward EPS/EBITDA/PE/sales) via OpenBB MCP tools.
"""

from langchain.agents import create_agent
from langchain_core.runnables import RunnableConfig

from ...model_config import ModelConfiguration
from ...prompts import render_template
from .utils import data_collection_middleware, get_tools

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


async def create_equity_estimates_data_collection_agent(config: RunnableConfig):
    """Build the equity estimates ReAct agent."""
    tools = await get_tools(config, MCP_TOOLS)
    prompt = render_template("data_collection/equity_estimates.jinja")
    model_config = ModelConfiguration.from_runnable_config(config)
    llm = model_config.get_llm()
    return create_agent(
        model=llm,
        tools=tools,
        system_prompt=prompt,
        middleware=data_collection_middleware(MCP_TOOLS),
    )
