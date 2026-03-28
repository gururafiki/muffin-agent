"""Options data collection agent.

ReAct agent that retrieves options chains with Greeks and implied volatility
surface data via OpenBB MCP tools.
"""

from langchain.agents import create_agent
from langchain_core.runnables import RunnableConfig

from ...model_config import ModelConfiguration
from ...prompts import render_template
from .utils import data_collection_middleware, get_tools

MCP_TOOLS = [
    "derivatives_options_chains",
    "derivatives_options_surface",
]


async def create_options_data_collection_agent(config: RunnableConfig):
    """Build the options ReAct agent."""
    tools = await get_tools(config, MCP_TOOLS)
    prompt = render_template("data_collection/options.jinja")
    model_config = ModelConfiguration.from_runnable_config(config)
    llm = model_config.get_llm()
    return create_agent(
        model=llm,
        tools=tools,
        system_prompt=prompt,
        middleware=data_collection_middleware(MCP_TOOLS),
    )
