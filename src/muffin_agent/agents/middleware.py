"""Shared agent middleware for tool result caching.

``ToolResultCacheMiddleware`` caches successful tool results via graph state
so duplicate calls within a single agent invocation return instantly.  All
state writes use ``Command(update=...)`` — no instance attributes are
mutated after ``__init__``.
"""

import json
import logging
import operator
from collections.abc import Awaitable, Callable
from typing import Annotated, Any, NotRequired

from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.types import Command

logger = logging.getLogger(__name__)


def cache_key(tool_call: dict) -> str:
    """Create a hashable key from tool name and sorted args."""
    args_json = json.dumps(tool_call.get("args", {}), sort_keys=True)
    return f"{tool_call['name']}:{args_json}"


class ToolResultCacheState(AgentState):
    """Extended state tracking cached successful tool results."""

    cached_tool_results: NotRequired[Annotated[dict[str, str], operator.or_]]


class ToolResultCacheMiddleware(AgentMiddleware["ToolResultCacheState"]):
    """Cache successful tool results via graph state, block duplicate calls.

    Uses ``Command(update=...)`` for state writes — never mutates instance
    attributes after ``__init__``.  Cache lives in graph state (thread-scoped,
    concurrency-safe).
    """

    state_schema = ToolResultCacheState

    def __init__(self, cacheable_tools: frozenset[str] | None = None) -> None:
        """Initialize with an optional whitelist of tool names to cache.

        Args:
            cacheable_tools: Immutable set of tool names to cache.  If
                ``None``, all tools are cached.  Use ``frozenset`` to make
                immutability explicit.
        """
        self.cacheable_tools = cacheable_tools  # set once, never mutated

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[
            [ToolCallRequest], Awaitable[ToolMessage | Command[Any]]
        ],
    ) -> ToolMessage | Command[Any]:
        """Return cached result on duplicate call, cache new successes."""
        tool_name = request.tool_call["name"]

        # Skip caching for non-whitelisted tools
        if (
            self.cacheable_tools is not None
            and tool_name not in self.cacheable_tools
        ):
            return await handler(request)

        key = cache_key(request.tool_call)

        # Read cache from graph state (immutable read)
        cached: dict[str, str] = request.state.get(
            "cached_tool_results", {}
        )

        if key in cached:
            logger.debug("Cache hit for tool '%s'", tool_name)
            return ToolMessage(
                content=f"[cached] {cached[key]}",
                tool_call_id=request.tool_call["id"],
            )

        result = await handler(request)

        # Cache successful results via Command (graph state update)
        if isinstance(result, ToolMessage) and not result.content.startswith(
            "Error"
        ):
            return Command(
                update={
                    "cached_tool_results": {key: result.content},
                    "messages": [result],
                }
            )

        return result
