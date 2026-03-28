"""News data collection agent.

ReAct agent that retrieves company news and global headlines with sentiment
signals via OpenBB MCP tools.
"""

from langchain.agents import create_agent
from langchain_core.runnables import RunnableConfig

from ...model_config import ModelConfiguration
from ...prompts import render_template
from .utils import data_collection_middleware, get_tools

MCP_TOOLS = [
    "news_company",
    "news_world",
]


async def create_news_data_collection_agent(config: RunnableConfig):
    """Build the news & sentiment ReAct agent."""
    tools = await get_tools(config, MCP_TOOLS)
    prompt = render_template("data_collection/news.jinja")
    model_config = ModelConfiguration.from_runnable_config(config)
    llm = model_config.get_llm()
    return create_agent(
        model=llm,
        tools=tools,
        system_prompt=prompt,
        middleware=data_collection_middleware(MCP_TOOLS),
    )
