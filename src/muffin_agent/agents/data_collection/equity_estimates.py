"""Equity Estimates data collection agent.

ReAct agent that retrieves analyst estimates data (price targets, consensus
estimates, forward EPS/EBITDA/PE/sales) via OpenBB MCP tools.
"""

from langchain_core.runnables import RunnableConfig

from ...model_config import ModelConfiguration
from ...utils.agent_builder import MuffinAgentBuilder
from .utils import get_tools

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
    llm = ModelConfiguration.from_runnable_config(config).get_llm()

    builder = (
        MuffinAgentBuilder(llm, name="equity_estimates")
        .with_system_prompt_template("data_collection/equity_estimates.jinja")
        .with_short_term_memory()
    )
    for tool in tools:
        builder = builder.with_tool(tool)
    return builder.build_react_agent()
