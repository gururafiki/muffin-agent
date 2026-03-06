"""Currency, commodity, and crypto data collection agent.

ReAct agent that retrieves FX rates, commodity spot prices, EIA energy
outlooks, and crypto price history via OpenBB MCP tools.
"""

from langchain.agents import create_agent

from ...config import Configuration
from ...prompts import render_template
from .utils import ToolErrorHandler, get_tools

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
    config: Configuration,
):
    """Build the currency, commodity, and crypto data ReAct agent."""
    tools = await get_tools(config, MCP_TOOLS)
    prompt = render_template("currency_commodities.jinja")
    llm = config.get_llm()
    return create_agent(
        model=llm,
        tools=tools,
        system_prompt=prompt,
        middleware=[ToolErrorHandler()],
    )
