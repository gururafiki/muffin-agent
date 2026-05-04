"""Fixed income and rates data collection agent.

ReAct agent that retrieves interest rates, yield curves, spreads, and bond
data via OpenBB MCP tools.
"""

from langchain_core.runnables import RunnableConfig

from ...model_config import ModelConfiguration
from ...utils.agent_builder import MuffinAgentBuilder
from .utils import get_tools

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


async def create_fixed_income_data_collection_agent(config: RunnableConfig):
    """Build the fixed income and rates ReAct agent."""
    tools = await get_tools(config, MCP_TOOLS)
    configuration = ModelConfiguration.from_runnable_config(config)
    primary, *fallbacks = configuration.get_llm_for_role("collector")
    summariser = configuration.get_summariser()

    builder = (
        MuffinAgentBuilder(primary, name="fixed_income")
        .with_system_prompt_template("data_collection/fixed_income.jinja")
        .with_fallback_models(*fallbacks)
        .with_short_term_memory()
    )
    if summariser is not None:
        builder = builder.with_tool_knowledge(summariser)
    for tool in tools:
        builder = builder.with_tool(tool)
    return builder.build_react_agent()
