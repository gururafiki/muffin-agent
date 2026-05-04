"""Equity Ownership & Short Interest data collection agent.

ReAct agent that retrieves ownership structure and short interest data
(major holders, institutional ownership, insider trades, 13F filings,
short interest/volume/FTDs) via OpenBB MCP tools.
"""

from langchain_core.runnables import RunnableConfig

from ...model_config import ModelConfiguration
from ...utils.agent_builder import MuffinAgentBuilder
from .utils import get_tools

MCP_TOOLS = [
    "equity_ownership_form_13f",
    "equity_ownership_government_trades",
    "equity_ownership_insider_trading",
    "equity_ownership_institutional",
    "equity_ownership_major_holders",
    "equity_ownership_share_statistics",
    "equity_shorts_fails_to_deliver",
    "equity_shorts_short_interest",
    "equity_shorts_short_volume",
]


async def create_equity_ownership_data_collection_agent(config: RunnableConfig):
    """Build the equity ownership & short interest ReAct agent."""
    tools = await get_tools(config, MCP_TOOLS)
    configuration = ModelConfiguration.from_runnable_config(config)
    primary, *fallbacks = configuration.get_llm_for_role("collector")
    summariser = configuration.get_summariser()

    builder = (
        MuffinAgentBuilder(primary, name="equity_ownership")
        .with_system_prompt_template("data_collection/equity_ownership.jinja")
        .with_fallback_models(*fallbacks)
        .with_short_term_memory()
    )
    if summariser is not None:
        builder = builder.with_tool_knowledge(summariser)
    for tool in tools:
        builder = builder.with_tool(tool)
    return builder.build_react_agent()
