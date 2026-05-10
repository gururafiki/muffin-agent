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

from .config import ToolResultCacheConfiguration

# Note: discover_cached_tool_outputs, get_tool_output_schema, and
# write_cached_tool_output_to_backend are not imported here — they are
# registered by MuffinAgentBuilder.with_sandbox() only.

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


def _should_skip_cache(result: ToolMessage, non_cacheable_patterns: list[str]) -> bool:
    """Return True when this ToolMessage result should not be written to the cache."""
    if getattr(result, "status", None) == "error":
        return True
    if isinstance(result.content, str):
        return is_error_content(result.content) or any(
            p in result.content.lower() for p in non_cacheable_patterns
        )
    return False  # list content with no error status → cacheable


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

        namespace = ("cache", tool_name)
        key = get_args_hash(args)

        # ── Cache hit ───────────────────────────────────────────────
        try:
            item = await store.aget(namespace, key)
            if item is not None and item.value.get("content"):
                cached = item.value["content"]
                logger.debug("Cache HIT for %s → %s/%s", tool_name, namespace, key)
                byte_size = (
                    len(cached)
                    if isinstance(cached, str)
                    else len(json.dumps(cached, default=str))
                )
                return ToolMessage(
                    content=cached,
                    tool_call_id=request.tool_call["id"],
                    additional_kwargs={
                        "cache": _cache_metadata(
                            hit=True,
                            tool_name=tool_name,
                            args_hash=key,
                            byte_size=byte_size,
                        )
                    },
                )
        except Exception:
            pass  # store read failed → treat as miss

        # ── Cache miss: execute tool ────────────────────────────────
        result = await handler(request)

        if not isinstance(result, ToolMessage):
            return result
        # Only cache str and list content (the two valid ToolMessage.content types).
        if not isinstance(result.content, (str, list)):
            return result

        cache_config = ToolResultCacheConfiguration.from_runnable_config(
            request.runtime.config
        )
        if _should_skip_cache(result, cache_config.non_cacheable_patterns):
            return result

        byte_size = (
            len(result.content)
            if isinstance(result.content, str)
            else len(json.dumps(result.content, default=str))
        )

        try:
            await store.aput(
                namespace,
                key,
                {
                    "content": result.content,
                    "tool_name": tool_name,
                    "args": args,
                    "cached_at": datetime.now(UTC).isoformat(),
                    "content_size": byte_size,
                },
            )
            logger.debug("Cache WRITE for %s → %s/%s", tool_name, namespace, key)
        except Exception:
            logger.debug(
                "Store write failed for %s, returning original result",
                tool_name,
                exc_info=True,
            )
            return result

        return ToolMessage(
            content=result.content,
            tool_call_id=result.tool_call_id,
            name=result.name,
            additional_kwargs={
                **result.additional_kwargs,
                "cache": _cache_metadata(
                    hit=False,
                    tool_name=tool_name,
                    args_hash=key,
                    byte_size=byte_size,
                ),
            },
        )
