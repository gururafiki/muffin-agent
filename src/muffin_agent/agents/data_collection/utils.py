"""Shared utilities for data collection agents."""

from deepagents.middleware.filesystem import FilesystemMiddleware
from langchain_core.runnables import RunnableConfig
from langchain_mcp_adapters.client import MultiServerMCPClient

from ...mcp_config import McpConfiguration
from ...middlewares import ToolErrorHandlerMiddleware, ToolResultCacheMiddleware
from ...sandbox import get_backend


async def get_tools(
    config: RunnableConfig,
    allowed_tools: list[str],
    custom_tools: list | None = None,
) -> list:
    """Load MCP tools filtered to *allowed_tools*, plus any custom tools.

    Skips the MCP connection entirely when *allowed_tools* is empty, which
    avoids an unnecessary network round-trip for agents that only use custom
    tools (e.g. the web-search agent).
    """
    if allowed_tools:
        mcp_config = McpConfiguration.from_runnable_config(config)
        client = MultiServerMCPClient(mcp_config.get_mcp_connections())
        all_tools = await client.get_tools()
        allowed = set(allowed_tools)
        mcp_tools = [t for t in all_tools if t.name in allowed]
    else:
        mcp_tools = []
    return mcp_tools + (custom_tools or [])


def data_collection_middleware(cacheable_tools: list[str]) -> list:
    """Build the standard middleware stack for data collection agents.

    Order: ToolErrorHandler (outer) → FilesystemMiddleware →
    ToolResultCacheMiddleware (inner).
    """
    return [
        ToolErrorHandlerMiddleware(),
        FilesystemMiddleware(backend=get_backend),
        ToolResultCacheMiddleware(
            cacheable_tools=frozenset(cacheable_tools),
        ),
    ]
