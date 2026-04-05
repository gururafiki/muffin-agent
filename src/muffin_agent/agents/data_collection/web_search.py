"""Web search and crawling data collection agent.

ReAct agent backed entirely by Firecrawl MCP tools. SearxNG is used as
Firecrawl's search engine (via ``SEARXNG_ENDPOINT`` in docker-compose) rather
than as a separate LangChain tool, keeping the tool surface minimal.
"""

from langchain.agents import create_agent
from langchain_core.runnables import RunnableConfig

from ...model_config import ModelConfiguration
from ...prompts import render_template
from .utils import data_collection_middleware, get_tools

# Firecrawl MCP tools to load — names must match what the MCP server exposes.
FIRECRAWL_MCP_TOOLS: list[str] = [
    "firecrawl_scrape",
    "firecrawl_crawl",
    "firecrawl_map",
    "firecrawl_search",
    "firecrawl_batch_scrape",
    "firecrawl_extract",
]


async def create_web_search_data_collection_agent(config: RunnableConfig):
    """Build the web search & crawling ReAct agent."""
    tools = await get_tools(config, allowed_tools=FIRECRAWL_MCP_TOOLS)
    prompt = render_template("data_collection/web_search.jinja")
    model_config = ModelConfiguration.from_runnable_config(config)
    return create_agent(
        model=model_config.get_llm(),
        tools=tools,
        system_prompt=prompt,
        middleware=data_collection_middleware(FIRECRAWL_MCP_TOOLS),
    )
