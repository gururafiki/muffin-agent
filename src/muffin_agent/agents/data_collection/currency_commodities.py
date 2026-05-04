"""Currency, commodity, and crypto data collection agent.

ReAct agent that retrieves FX rates, commodity spot prices, EIA energy
outlooks, and crypto price history via OpenBB MCP tools.
"""

from langchain_core.runnables import RunnableConfig

from ...model_config import ModelConfiguration
from ...utils.agent_builder import MuffinAgentBuilder
from .utils import get_tools

MCP_TOOLS = [
    "commodity_petroleum_status_report",
    "commodity_price_spot",
    "commodity_short_term_energy_outlook",
    "crypto_price_historical",
    "crypto_search",
    "currency_price_historical",
    "currency_reference_rates",
    "currency_search",
    "currency_snapshots",
]


async def create_currency_commodities_data_collection_agent(
    config: RunnableConfig,
):
    """Build the currency, commodity, and crypto data ReAct agent."""
    tools = await get_tools(config, MCP_TOOLS)
    configuration = ModelConfiguration.from_runnable_config(config)
    primary, *fallbacks = configuration.get_llm_for_role("collector")
    summariser = configuration.get_summariser()

    builder = (
        MuffinAgentBuilder(primary, name="currency_commodities")
        .with_system_prompt_template("data_collection/currency_commodities.jinja")
        .with_fallback_models(*fallbacks)
        .with_short_term_memory()
    )
    if summariser is not None:
        builder = builder.with_tool_knowledge(summariser)
    for tool in tools:
        builder = builder.with_tool(tool)
    return builder.build_react_agent()
