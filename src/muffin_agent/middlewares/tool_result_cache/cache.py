"""Shared cache primitives for the tool result cache.

Single source of truth for hash + namespace logic so both
:class:`ToolResultCacheMiddleware` (LLM-driven tool calls) and direct
callers (deterministic graph nodes that bypass the ReAct loop) hit the
same cache keys.

Cache layout (unchanged from the original middleware):

* Namespace: ``("cache", tool_name)``
* Key: ``get_args_hash(args)`` — sha256 of sorted-JSON args, first 12 hex chars
* Value: ``{"content": str | list, "tool_name": str, "args": dict,
            "cached_at": ISO8601 str, "content_size": int}``

Three public APIs:

* :func:`cache_lookup` — HIT check, returns ``(hit, content)``
* :func:`cache_store` — MISS write, returns ``True`` on success
* :func:`cached_invoke` — high-level HIT/MISS wrapper around
  ``tool.ainvoke(args)`` for non-middleware callers

Plus the two foundational helpers (:func:`get_args_hash`,
:func:`is_error_content`) used by the middleware as well.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool
from langgraph.store.base import BaseStore

logger = logging.getLogger(__name__)


def get_args_hash(args: dict[str, Any]) -> str:
    """Build a deterministic 12-char hex hash from sorted args."""
    args_json = json.dumps(args, sort_keys=True)
    return hashlib.sha256(args_json.encode()).hexdigest()[:12]


def is_error_content(content: object) -> bool:
    """Check whether a tool message content string contains an error."""
    if not isinstance(content, str):
        return False
    lower = content.lower()
    return lower.startswith("error") or lower.startswith("duplicate call blocked")


def _content_size(content: Any) -> int:
    """Return the byte size of *content* for cache metadata."""
    if isinstance(content, str):
        return len(content)
    return len(json.dumps(content, default=str))


def is_content_cacheable(
    content: Any,
    non_cacheable_patterns: Sequence[str] = (),
) -> bool:
    """Return True if *content* should be written to the cache.

    Skips:

    * Non-``str``/``list`` content (None, dicts, etc.)
    * Error-flagged strings (matched by :func:`is_error_content`)
    * Strings containing any *non_cacheable_patterns* substring (case-insensitive)

    ``list`` content (e.g. Firecrawl multi-part results) with no error status
    is always cacheable.
    """
    if isinstance(content, str):
        if is_error_content(content):
            return False
        lower = content.lower()
        return not any(p in lower for p in non_cacheable_patterns)
    if isinstance(content, list):
        return True
    return False


async def cache_lookup(
    tool_name: str,
    args: dict[str, Any],
    store: BaseStore,
) -> tuple[bool, Any]:
    """Look up a cached tool result.

    Args:
        tool_name: Tool name (used as the second tuple element of the namespace).
        args: Tool args dict; hashed via :func:`get_args_hash`.
        store: The shared ``BaseStore``.

    Returns:
        ``(hit, content)`` — ``hit=True`` with the cached content on a successful
        HIT; ``(False, None)`` on MISS, store error, or empty/missing content.
    """
    namespace = ("cache", tool_name)
    key = get_args_hash(args)
    try:
        item = await store.aget(namespace, key)
    except Exception:
        logger.debug("cache_lookup: store read failed for %s/%s", namespace, key)
        return False, None
    if item is None:
        return False, None
    content = item.value.get("content")
    if not content:
        return False, None
    logger.debug("Cache HIT for %s → %s/%s", tool_name, namespace, key)
    return True, content


async def cache_store(
    tool_name: str,
    args: dict[str, Any],
    store: BaseStore,
    content: Any,
    *,
    non_cacheable_patterns: Sequence[str] = (),
) -> bool:
    """Write *content* to the cache.

    Args:
        tool_name: Tool name.
        args: Tool args dict.
        store: The shared ``BaseStore``.
        content: Raw tool result (``str`` or ``list``).
        non_cacheable_patterns: Case-insensitive substrings; matches are skipped.

    Returns:
        ``True`` on successful write; ``False`` on skip-pattern, non-cacheable
        content type, or store error.
    """
    if not is_content_cacheable(content, non_cacheable_patterns):
        return False
    namespace = ("cache", tool_name)
    key = get_args_hash(args)
    try:
        await store.aput(
            namespace,
            key,
            {
                "content": content,
                "tool_name": tool_name,
                "args": args,
                "cached_at": datetime.now(UTC).isoformat(),
                "content_size": _content_size(content),
            },
        )
    except Exception:
        logger.debug(
            "cache_store: store write failed for %s/%s", namespace, key, exc_info=True
        )
        return False
    logger.debug("Cache WRITE for %s → %s/%s", tool_name, namespace, key)
    return True


async def cached_invoke(
    tool: BaseTool,
    args: dict[str, Any],
    store: BaseStore | None,
    *,
    tool_name: str | None = None,
    non_cacheable_patterns: Sequence[str] = (),
) -> Any:
    """Invoke *tool* with HIT short-circuit + MISS write.

    Intended for non-LLM-driven callers (deterministic graph nodes that
    bypass the ReAct loop and ``ToolResultCacheMiddleware``). Cache keys
    collide perfectly with the middleware-driven path: namespace
    ``("cache", tool_name)`` + ``get_args_hash(args)``.

    Args:
        tool: ``BaseTool`` to invoke.
        args: Tool args dict, passed verbatim to ``tool.ainvoke(args)``.
        store: Shared ``BaseStore`` for cache.  ``None`` bypasses cache.
        tool_name: Override for ``tool.name`` (rarely needed).
        non_cacheable_patterns: See :func:`cache_store`.

    Returns:
        Raw tool result content (``str`` or ``list`` — the
        ``ToolMessage.content`` shape).  If the tool returns a
        non-``ToolMessage`` result, it is returned as-is and cached when
        possible.
    """
    name = tool_name or tool.name
    if store is not None:
        hit, cached = await cache_lookup(name, args, store)
        if hit:
            return cached

    result = await tool.ainvoke(args)
    content = result.content if isinstance(result, ToolMessage) else result

    if store is not None:
        await cache_store(
            name, args, store, content, non_cacheable_patterns=non_cacheable_patterns
        )

    return content
