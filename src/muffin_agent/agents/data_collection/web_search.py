"""Web search and crawling data collection agent.

ReAct agent that searches the web via SearxNG and scrapes URLs via Firecrawl,
with MarkItDown for document conversion. Uses custom LangChain tools rather
than OpenBB MCP tools.
"""

from langchain.agents import create_agent
from langchain_core.runnables import RunnableConfig

from ...model_config import ModelConfiguration
from ...prompts import render_template
from ...tools.web import convert_document, web_crawl, web_map, web_scrape, web_search
from .utils import data_collection_middleware, get_tools

# No OpenBB MCP tools — this agent uses direct HTTP tools only.
MCP_TOOLS: list[str] = []

WEB_TOOLS = [web_search, web_scrape, web_crawl, web_map, convert_document]


async def create_web_search_data_collection_agent(config: RunnableConfig):
    """Build the web search & crawling ReAct agent."""
    tools = await get_tools(config, MCP_TOOLS, custom_tools=WEB_TOOLS)
    prompt = render_template("data_collection/web_search.jinja")
    model_config = ModelConfiguration.from_runnable_config(config)
    llm = model_config.get_llm()
    return create_agent(
        model=llm,
        tools=tools,
        system_prompt=prompt,
        middleware=data_collection_middleware([]),
    )
