"""News data collection agent.

ReAct agent that retrieves company news and global headlines with sentiment
signals via OpenBB MCP tools.
"""

from langchain_core.runnables import RunnableConfig

from ...model_config import ModelConfiguration
from ...utils.agent_builder import MuffinAgentBuilder
from .utils import get_tools

MCP_TOOLS = [
    "news_company",
    "news_world",
]


async def create_news_data_collection_agent(config: RunnableConfig):
    """Build the news & sentiment ReAct agent."""
    tools = await get_tools(config, MCP_TOOLS)
    llm = ModelConfiguration.from_runnable_config(config).get_llm()

    builder = (
        MuffinAgentBuilder(llm, name="news")
        .with_system_prompt_template("data_collection/news.jinja")
        .with_short_term_memory()
    )
    for tool in tools:
        builder = builder.with_tool(tool)
    return builder.build_react_agent()
