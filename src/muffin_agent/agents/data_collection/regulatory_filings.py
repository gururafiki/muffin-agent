"""Regulatory and filings data collection agent.

ReAct agent that retrieves SEC filings, CFTC Commitment of Traders reports,
and US congressional bill data via OpenBB MCP tools.
"""

from langchain.agents import create_agent

from ...config import Configuration
from ...prompts import render_template
from .utils import ToolErrorHandler, get_tools

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


async def create_regulatory_filings_data_collection_agent(config: Configuration):
    """Build the regulatory filings ReAct agent."""
    tools = await get_tools(config, MCP_TOOLS)
    prompt = render_template("regulatory_filings.jinja")
    llm = config.get_llm()
    return create_agent(
        model=llm,
        tools=tools,
        system_prompt=prompt,
        middleware=[ToolErrorHandler()],
    )
