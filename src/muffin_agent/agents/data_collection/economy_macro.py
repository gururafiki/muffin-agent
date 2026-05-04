"""Economy and macro data collection agent.

ReAct agent that retrieves macroeconomic indicators, FRED series, economic
surveys, and shipping data via OpenBB MCP tools.
"""

from langchain_core.runnables import RunnableConfig

from ...model_config import ModelConfiguration
from ...utils.agent_builder import MuffinAgentBuilder
from .utils import get_tools

MCP_TOOLS = [
    "economy_available_indicators",
    "economy_balance_of_payments",
    "economy_calendar",
    "economy_central_bank_holdings",
    "economy_composite_leading_indicator",
    "economy_country_profile",
    "economy_cpi",
    "economy_direction_of_trade",
    "economy_export_destinations",
    "economy_fomc_documents",
    "economy_fred_regional",
    "economy_fred_release_table",
    "economy_fred_search",
    "economy_fred_series",
    "economy_gdp_forecast",
    "economy_gdp_nominal",
    "economy_gdp_real",
    "economy_house_price_index",
    "economy_indicators",
    "economy_interest_rates",
    "economy_money_measures",
    "economy_pce",
    "economy_primary_dealer_fails",
    "economy_primary_dealer_positioning",
    "economy_retail_prices",
    "economy_risk_premium",
    "economy_share_price_index",
    "economy_shipping_chokepoint_info",
    "economy_shipping_chokepoint_volume",
    "economy_shipping_port_info",
    "economy_shipping_port_volume",
    "economy_survey_bls_search",
    "economy_survey_bls_series",
    "economy_survey_economic_conditions_chicago",
    "economy_survey_manufacturing_outlook_ny",
    "economy_survey_manufacturing_outlook_texas",
    "economy_survey_nonfarm_payrolls",
    "economy_survey_sloos",
    "economy_survey_university_of_michigan",
    "economy_unemployment",
]


async def create_economy_macro_data_collection_agent(config: RunnableConfig):
    """Build the economy and macro ReAct agent."""
    tools = await get_tools(config, MCP_TOOLS)
    configuration = ModelConfiguration.from_runnable_config(config)
    primary, *fallbacks = configuration.get_llm_for_role("collector")
    summariser = configuration.get_summariser()

    builder = (
        MuffinAgentBuilder(primary, name="economy_macro")
        .with_system_prompt_template("data_collection/economy_macro.jinja")
        .with_fallback_models(*fallbacks)
        .with_short_term_memory()
    )
    if summariser is not None:
        builder = builder.with_tool_knowledge(summariser)
    for tool in tools:
        builder = builder.with_tool(tool)
    return builder.build_react_agent()
