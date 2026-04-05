"""Web search and crawling data collection agent.

ReAct agent combining:
- SearxNG meta-search via LangChain ``SearxSearchResults`` (lightweight
  snippets with engine/category metadata, direct JSON output).
- Firecrawl MCP tools for scraping, crawling, URL discovery, batch scraping,
  and structured LLM extraction.
"""

from langchain.agents import create_agent
from langchain_community.agent_toolkits import load_tools
from langchain_core.runnables import RunnableConfig

from ...model_config import ModelConfiguration
from ...prompts import render_template
from ...web_config import WebConfiguration
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
    web_cfg = WebConfiguration.from_runnable_config(config)

    # SearxNG meta-search: structured JSON results (snippet, title, link,
    # engines, category) with native async support.
    searxng_tools = load_tools(
        ["searx-search-results-json"],
        searx_host=web_cfg.searxng_url,
        num_results=10,
    )

    # Firecrawl MCP tools (scrape, crawl, map, search, batch_scrape, extract).
    tools = await get_tools(
        config,
        allowed_tools=FIRECRAWL_MCP_TOOLS,
        custom_tools=[*searxng_tools],
    )

    prompt = render_template("data_collection/web_search.jinja")
    model_config = ModelConfiguration.from_runnable_config(config)
    # data_collection_middleware([]) passes an empty cacheable_tools list,
    # disabling ToolResultCacheMiddleware for all tools in this agent.
    # Unlike OpenBB data agents (which pass their MCP tool list so identical
    # API calls are deduplicated across agents in the same run), web results
    # are intentionally not cached: the same URL or query may return different
    # content on each call, and cross-agent deduplication has little value for
    # ephemeral search/scrape operations.
    return create_agent(
        model=model_config.get_llm(),
        tools=tools,
        system_prompt=prompt,
        middleware=data_collection_middleware([]),
    )
