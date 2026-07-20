"""Shared middleware for agent tool call interception.

``ToolResultCacheMiddleware`` caches successful tool results in a shared
``InMemoryStore`` so that identical calls across different agents are
deduplicated automatically.

**Strict content invariant** — the middleware never mutates
``ToolMessage.content``. Cache provenance lives in
``additional_kwargs["cache"]`` so downstream consumers that
``json.loads(message.content)`` keep working.

Supports both ``str`` and ``list`` content (the two valid
``ToolMessage.content`` types per LangChain). List content (returned by
Firecrawl/MCP tools) round-trips losslessly through both ``InMemoryStore``
and JSON-serialising persistent stores (Postgres) since ``list[str | dict]``
is natively JSON-serialisable.

Size-based offloading of oversized tool results is *not* this middleware's
job: ``deepagents.middleware.filesystem.FilesystemMiddleware`` already
evicts large results to ``{root}/large_tool_results/<tool_call_id>``
(default threshold 20K tokens). Offload messages are excluded from the
cache (see ``non_cacheable_patterns``) because they embed an ephemeral
path keyed by ``tool_call_id`` that becomes stale on a cache hit.

Cache primitives live in :mod:`muffin_agent.middlewares.tool_result_cache.cache`
(``cache_lookup``, ``cache_store``, ``get_args_hash``, ``is_error_content``).
The middleware composes them so direct callers (e.g. specialist Python nodes
calling ``cached_invoke``) and LLM-driven callers share one hashing/namespace
scheme.
"""

import json
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.types import Command

from .cache import (
    cache_lookup,
    cache_store,
    get_args_hash,  # re-exported below for backwards compatibility
    is_error_content,  # re-exported below for backwards compatibility
)
from .config import ToolResultCacheConfiguration

# Note: discover_cached_tool_outputs, get_tool_output_schema, and
# write_cached_tool_output_to_backend are not imported here — they are
# registered by MuffinAgentBuilder.with_sandbox() only.

logger = logging.getLogger(__name__)

# Re-exported for backwards compatibility — original public API of this module.
__all__ = [
    "ToolResultCacheMiddleware",
    "get_args_hash",
    "is_error_content",
]


def _cache_metadata(
    *,
    hit: bool,
    tool_name: str,
    args_hash: str,
    byte_size: int,
) -> dict[str, Any]:
    """Build the ``additional_kwargs['cache']`` metadata payload."""
    return {
        "hit": hit,
        "tool_name": tool_name,
        "args_hash": args_hash,
        "byte_size": byte_size,
    }


def _content_byte_size(content: Any) -> int:
    """Compute the cache-metadata byte size (matches cache.py helper)."""
    if isinstance(content, str):
        return len(content)
    return len(json.dumps(content, default=str))


class ToolResultCacheMiddleware(AgentMiddleware):
    """Cache tool results in a shared store for cross-agent reuse.

    Uses ``ToolRuntime.store`` (``InMemoryStore``) as the cache backend.
    All agents sharing the same store see each other's cached results,
    enabling cross-agent deduplication of MCP and computation tool calls.

    Cache tools (``discover_cached_tool_outputs``, ``get_tool_output_schema``,
    ``write_cached_tool_output_to_backend``) are **not** registered here —
    they are scoped to sandbox-enabled agents by ``MuffinAgentBuilder.with_sandbox()``.

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
        # Cache tools (discover_cached_tool_outputs, get_tool_output_schema,
        # write_cached_tool_output_to_backend) are registered by
        # MuffinAgentBuilder.with_sandbox() — they require sandbox routing.
        self.tools: list[Any] = []

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

        args_hash = get_args_hash(args)

        # ── Cache hit ───────────────────────────────────────────────
        hit, cached = await cache_lookup(tool_name, args, store)
        if hit:
            return ToolMessage(
                content=cached,
                tool_call_id=request.tool_call["id"],
                additional_kwargs={
                    "cache": _cache_metadata(
                        hit=True,
                        tool_name=tool_name,
                        args_hash=args_hash,
                        byte_size=_content_byte_size(cached),
                    )
                },
            )

        # ── Cache miss: execute tool ────────────────────────────────
        result = await handler(request)

        if not isinstance(result, ToolMessage):
            return result
        # `status` is the authoritative success/failure signal — never cache
        # (or drop it when reconstructing) an errored result. Content-string
        # heuristics like `is_error_content()` are a fallback for messages
        # that never set `status`, not a substitute for checking it first.
        if result.status == "error":
            return result
        # Only cache str and list content (the two valid ToolMessage.content types).
        if not isinstance(result.content, (str, list)):
            return result

        cache_config = ToolResultCacheConfiguration.from_runnable_config(
            request.runtime.config
        )
        written = await cache_store(
            tool_name,
            args,
            store,
            result.content,
            non_cacheable_patterns=cache_config.non_cacheable_patterns,
        )
        if not written:
            return result

        return ToolMessage(
            content=result.content,
            tool_call_id=result.tool_call_id,
            name=result.name,
            status=result.status,
            additional_kwargs={
                **result.additional_kwargs,
                "cache": _cache_metadata(
                    hit=False,
                    tool_name=tool_name,
                    args_hash=args_hash,
                    byte_size=_content_byte_size(result.content),
                ),
            },
        )
