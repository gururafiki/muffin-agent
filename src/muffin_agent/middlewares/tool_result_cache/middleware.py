"""Shared middleware for agent tool call interception.

``ToolResultCacheMiddleware`` caches successful tool results in a shared
``InMemoryStore`` so that identical calls across different agents are
deduplicated automatically.
"""

import hashlib
import json
import logging
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.types import Command

from .tools import (
    discover_cached_data,
    get_tool_output_schema,
    write_tool_output_to_backend,
)

logger = logging.getLogger(__name__)


def get_args_hash(args: dict[str, Any]) -> str:
    """Build a deterministic 12-char hex hash from sorted args."""
    args_json = json.dumps(args, sort_keys=True)
    return hashlib.sha256(args_json.encode()).hexdigest()[:12]


def is_error_content(content: str) -> bool:
    """Check whether a tool message contains an error."""
    if not isinstance(content, str):
        return False
    lower = content.lower()
    return lower.startswith("error") or lower.startswith("duplicate call blocked")


class ToolResultCacheMiddleware(AgentMiddleware):
    """Cache tool results in a shared store for cross-agent reuse.

    Uses ``ToolRuntime.store`` (``InMemoryStore``) as the cache backend.
    All agents sharing the same store see each other's cached results,
    enabling cross-agent deduplication of MCP and computation tool calls.

    Args:
        cacheable_tools: Tool names to cache.  ``None`` caches every tool;
            a ``frozenset`` restricts caching to the listed names.
    """

    def __init__(
        self,
        cacheable_tools: frozenset[str] | None = None,
    ) -> None:
        """Initialize with an optional tool whitelist."""
        self._cacheable_tools = cacheable_tools
        self.tools = [
            discover_cached_data,
            get_tool_output_schema,
            write_tool_output_to_backend,
        ]

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        """Intercept tool calls to cache/return cached results."""
        tool_name: str = request.tool_call["name"]

        # Skip non-cacheable tools.
        if self._cacheable_tools is not None and tool_name not in self._cacheable_tools:
            return await handler(request)

        args: dict[str, Any] = request.tool_call.get("args", {})
        store = request.runtime.store
        if store is None:
            return await handler(request)

        namespace = ("cache", tool_name)
        key = get_args_hash(args)

        # ── Cache hit ───────────────────────────────────────────────
        try:
            item = await store.aget(namespace, key)
            if item is not None and item.value.get("content"):
                cached = item.value["content"]
                logger.debug("Cache HIT for %s → %s/%s", tool_name, namespace, key)
                return ToolMessage(
                    content=(
                        f"[Cached result — tool: {tool_name}, "
                        f"args_hash: {key}]\n\n{cached}"
                    ),
                    tool_call_id=request.tool_call["id"],
                )
        except Exception:
            pass  # store read failed → treat as miss

        # ── Cache miss: execute tool ────────────────────────────────
        result = await handler(request)

        # Only cache successful ToolMessage results with string content.
        if (
            isinstance(result, ToolMessage)
            and isinstance(result.content, str)
            and not is_error_content(result.content)
        ):
            try:
                await store.aput(
                    namespace,
                    key,
                    {
                        "content": result.content,
                        "tool_name": tool_name,
                        "args": args,
                        "cached_at": datetime.now(UTC).isoformat(),
                        "content_size": len(result.content),
                    },
                )
                logger.debug("Cache WRITE for %s → %s/%s", tool_name, namespace, key)
                result = ToolMessage(
                    content=(
                        f"{result.content}\n\n"
                        f"[Data cached — tool: {tool_name}, args_hash: {key}]"
                    ),
                    tool_call_id=result.tool_call_id,
                    name=result.name,
                )
            except Exception:
                logger.debug(
                    "Store write failed for %s, returning original result",
                    tool_name,
                    exc_info=True,
                )

        return result
