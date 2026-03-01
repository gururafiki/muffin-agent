"""Shared utilities for data collection agents."""

import json
import operator
from collections.abc import Awaitable, Callable
from typing import Annotated, Any, NotRequired

from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import ToolMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.types import Command

from ...config import Configuration

# Error substrings that indicate permanent failures (no point retrying).
PERMANENT_ERROR_PATTERNS = (
    "missing credential",
    "api_key",
    "authentication",
    "unauthorized",
    "403",
    "404",
    "not found",
    "not supported",
    "invalid parameter",
)


def _is_permanent_error(error_msg: str) -> bool:
    """Check if error message matches a known permanent failure pattern."""
    lower = error_msg.lower()
    return any(pattern in lower for pattern in PERMANENT_ERROR_PATTERNS)


def _cache_key(tool_call: dict) -> str:
    """Create a hashable key from tool name and sorted args."""
    args_json = json.dumps(tool_call.get("args", {}), sort_keys=True)
    return f"{tool_call['name']}:{args_json}"


class ToolErrorState(AgentState):
    """Extended agent state that tracks permanently failed tool calls."""

    failed_tool_calls: NotRequired[Annotated[dict[str, str], operator.or_]]


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


class ToolErrorHandler(AgentMiddleware["ToolErrorState"]):
    """Middleware that catches tool errors and blocks duplicate permanent failures.

    Tracks (tool_name, args) pairs that produced permanent errors in graph
    state and short-circuits duplicate calls with cached error messages.
    """

    state_schema = ToolErrorState

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        """Catch tool exceptions, cache permanent failures, block duplicates."""
        key = _cache_key(request.tool_call)
        failed_calls: dict[str, str] = request.state.get("failed_tool_calls", {})

        if key in failed_calls:
            return ToolMessage(
                content=(
                    f"DUPLICATE CALL BLOCKED: This tool was already called with "
                    f"identical arguments and failed permanently. "
                    f"Previous error: {failed_calls[key]}"
                ),
                tool_call_id=request.tool_call["id"],
            )

        try:
            return await handler(request)
        except Exception as e:
            error_msg = str(e)
            if _is_permanent_error(error_msg):
                updated_failures = {**failed_calls, key: error_msg}
                return Command(
                    update={
                        "failed_tool_calls": updated_failures,
                        "messages": [
                            ToolMessage(
                                content=f"Error (permanent): {error_msg}",
                                tool_call_id=request.tool_call["id"],
                            )
                        ],
                    }
                )
            return ToolMessage(
                content=f"Error: {error_msg}",
                tool_call_id=request.tool_call["id"],
            )
