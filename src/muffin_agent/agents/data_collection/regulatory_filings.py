"""Regulatory and filings data collection agent.

ReAct agent that retrieves SEC filings, CFTC Commitment of Traders reports,
and US congressional bill data via OpenBB MCP tools.
"""

from langchain_core.runnables import RunnableConfig

from ...model_config import ModelConfiguration
from ...utils.agent_builder import MuffinAgentBuilder
from .utils import get_tools

MCP_TOOLS = [
    "regulators_cftc_cot",
    "regulators_cftc_cot_search",
    "regulators_sec_cik_map",
    "regulators_sec_filing_headers",
    "regulators_sec_htm_file",
    "regulators_sec_institutions_search",
    "regulators_sec_rss_litigation",
    "regulators_sec_schema_files",
    "regulators_sec_sic_search",
    "regulators_sec_symbol_map",
    "uscongress_bill_info",
    "uscongress_bill_text",
    "uscongress_bill_text_urls",
    "uscongress_bills",
]


async def create_regulatory_filings_data_collection_agent(config: RunnableConfig):
    """Build the regulatory filings ReAct agent."""
    tools = await get_tools(config, MCP_TOOLS)
    configuration = ModelConfiguration.from_runnable_config(config)
    primary, *fallbacks = configuration.get_llm_for_role("collector")
    summariser = configuration.get_summariser()

    builder = (
        MuffinAgentBuilder(primary, name="regulatory_filings")
        .with_system_prompt_template("data_collection/regulatory_filings.jinja")
        .with_fallback_models(*fallbacks)
        .with_short_term_memory()
    )
    if summariser is not None:
        builder = builder.with_tool_knowledge(summariser)
    for tool in tools:
        builder = builder.with_tool(tool)
    return builder.build_react_agent()
