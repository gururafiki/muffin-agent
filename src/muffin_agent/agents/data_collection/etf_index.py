"""ETF and index data collection agent.

ReAct agent that retrieves ETF info, sector weights, holdings, index levels,
and S&P 500 valuation multiples via OpenBB MCP tools.
"""

from langchain_core.runnables import RunnableConfig

from ...model_config import ModelConfiguration
from ...utils.agent_builder import MuffinAgentBuilder
from .utils import get_tools

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


async def create_etf_index_data_collection_agent(config: RunnableConfig):
    """Build the ETF and index data ReAct agent."""
    tools = await get_tools(config, MCP_TOOLS)
    configuration = ModelConfiguration.from_runnable_config(config)
    primary, *fallbacks = configuration.get_llm_for_role("collector")
    summariser = configuration.get_summariser()

    builder = (
        MuffinAgentBuilder(primary, name="etf_index")
        .with_system_prompt_template("data_collection/etf_index.jinja")
        .with_fallback_models(*fallbacks)
        .with_short_term_memory()
    )
    if summariser is not None:
        builder = builder.with_tool_knowledge(summariser)
    for tool in tools:
        builder = builder.with_tool(tool)
    return builder.build_react_agent()
