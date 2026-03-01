"""Options data collection agent.

ReAct agent that retrieves options chains with Greeks and implied volatility
surface data via OpenBB MCP tools.
"""

from langchain.agents import create_agent

from ...config import Configuration
from ...prompts import render_template
from .utils import ToolErrorHandler, get_tools

MCP_TOOLS = [
    "derivatives_options_chains",
    "derivatives_options_surface",
]


async def create_options_data_collection_agent(config: Configuration):
    """Build the options ReAct agent."""
    tools = await get_tools(config, MCP_TOOLS)
    prompt = render_template("options.jinja")
    llm = config.get_llm()
    return create_agent(
        model=llm,
        tools=tools,
        system_prompt=prompt,
        middleware=[ToolErrorHandler()],
    )
