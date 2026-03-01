"""News data collection agent.

ReAct agent that retrieves company news and global headlines with sentiment
signals via OpenBB MCP tools.
"""

from langchain.agents import create_agent

from ...config import Configuration
from ...prompts import render_template
from .utils import ToolErrorHandler, get_tools

MCP_TOOLS = [
    "news_company",
    "news_world",
]


async def create_news_data_collection_agent(config: Configuration):
    """Build the news & sentiment ReAct agent."""
    tools = await get_tools(config, MCP_TOOLS)
    prompt = render_template("news.jinja")
    llm = config.get_llm()
    return create_agent(
        model=llm,
        tools=tools,
        system_prompt=prompt,
        middleware=[ToolErrorHandler()],
    )
