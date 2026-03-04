"""Fixed income and rates data collection agent.

ReAct agent that retrieves interest rates, yield curves, spreads, and bond
data via OpenBB MCP tools.
"""

from langchain.agents import create_agent

from ...config import Configuration
from ...prompts import render_template
from .utils import ToolErrorHandler, get_tools

MCP_TOOLS = [
    "fixedincome_bond_indices",
    "fixedincome_corporate_bond_prices",
    "fixedincome_corporate_commercial_paper",
    "fixedincome_corporate_hqm",
    "fixedincome_corporate_spot_rates",
    "fixedincome_government_tips_yields",
    "fixedincome_government_treasury_auctions",
    "fixedincome_government_treasury_prices",
    "fixedincome_government_treasury_rates",
    "fixedincome_government_yield_curve",
    "fixedincome_mortgage_indices",
    "fixedincome_rate_ameribor",
    "fixedincome_rate_dpcredit",
    "fixedincome_rate_ecb",
    "fixedincome_rate_effr",
    "fixedincome_rate_effr_forecast",
    "fixedincome_rate_estr",
    "fixedincome_rate_iorb",
    "fixedincome_rate_overnight_bank_funding",
    "fixedincome_rate_sofr",
    "fixedincome_rate_sonia",
    "fixedincome_spreads_tcm",
    "fixedincome_spreads_tcm_effr",
    "fixedincome_spreads_treasury_effr",
]


async def create_fixed_income_data_collection_agent(config: Configuration):
    """Build the fixed income and rates ReAct agent."""
    tools = await get_tools(config, MCP_TOOLS)
    prompt = render_template("fixed_income.jinja")
    llm = config.get_llm()
    return create_agent(
        model=llm,
        tools=tools,
        system_prompt=prompt,
        middleware=[ToolErrorHandler()],
    )
