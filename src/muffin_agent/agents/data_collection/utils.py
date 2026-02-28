"""Shared utilities for data collection agents."""

from langchain.agents.middleware import wrap_tool_call
from langchain_core.messages import ToolMessage
from langchain_mcp_adapters.client import MultiServerMCPClient

from ...config import Configuration


async def get_tools(
    config: Configuration,
    allowed_tools: list[str],
    custom_tools: list | None = None,
) -> list:
    """Load MCP tools filtered to *allowed_tools*, plus any custom tools."""
    client = MultiServerMCPClient(config.get_mcp_connections())
    all_tools = await client.get_tools()
    allowed = set(allowed_tools)
    mcp_tools = [t for t in all_tools if t.name in allowed]
    return mcp_tools + (custom_tools or [])


@wrap_tool_call
async def handle_tool_errors(request, handler):
    """Catch tool exceptions and return error messages to the agent."""
    try:
        return await handler(request)
    except Exception as e:
        return ToolMessage(
            content=f"Error: {e!s}",
            tool_call_id=request.tool_call["id"],
        )
