"""Currency, commodity, and crypto data collection agent.

ReAct agent that retrieves FX rates, commodity spot prices, EIA energy
outlooks, and crypto price history via OpenBB MCP tools.
"""

from langchain.agents import create_agent
from langchain_core.runnables import RunnableConfig

from ...model_config import ModelConfiguration
from ...prompts import render_template
from .utils import data_collection_middleware, get_tools

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
    prompt = render_template("data_collection/currency_commodities.jinja")
    model_config = ModelConfiguration.from_runnable_config(config)
    llm = model_config.get_llm()
    return create_agent(
        model=llm,
        tools=tools,
        system_prompt=prompt,
        middleware=data_collection_middleware(MCP_TOOLS),
    )
