"""Web search and crawling data collection agent.

ReAct agent backed entirely by Firecrawl MCP tools. SearxNG is used as
Firecrawl's search engine (via ``SEARXNG_ENDPOINT`` in docker-compose) rather
than as a separate LangChain tool, keeping the tool surface minimal.
"""

from langchain_core.runnables import RunnableConfig

from ...model_config import ModelConfiguration
from ...utils.agent_builder import MuffinAgentBuilder
from .utils import get_tools

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
    configuration = ModelConfiguration.from_runnable_config(config)
    primary, *fallbacks = configuration.get_llm_for_role("collector")
    summariser = configuration.get_summariser()

    builder = (
        MuffinAgentBuilder(primary, name="web_search")
        .with_system_prompt_template("data_collection/web_search.jinja")
        .with_fallback_models(*fallbacks)
        .with_short_term_memory()
    )
    if summariser is not None:
        builder = builder.with_tool_knowledge(summariser)
    for tool in tools:
        builder = builder.with_tool(tool)
    return builder.build_react_agent()
