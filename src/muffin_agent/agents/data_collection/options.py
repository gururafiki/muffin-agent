"""Options data collection agent.

ReAct agent that retrieves options chains with Greeks and implied volatility
surface data via OpenBB MCP tools.
"""

from langchain_core.runnables import RunnableConfig

from ...model_config import ModelConfiguration
from ...utils.agent_builder import MuffinAgentBuilder
from .utils import get_tools

MCP_TOOLS = [
    "derivatives_options_chains",
    "derivatives_options_surface",
]


async def create_options_data_collection_agent(config: RunnableConfig):
    """Build the options ReAct agent."""
    tools = await get_tools(config, MCP_TOOLS)
    llm = ModelConfiguration.from_runnable_config(config).get_llm()

    builder = (
        MuffinAgentBuilder(llm, name="options")
        .with_system_prompt_template("data_collection/options.jinja")
        .with_short_term_memory()
    )
    for tool in tools:
        builder = builder.with_tool(tool)
    return builder.build_react_agent()
