"""Generic store CRUD tools for LangGraph agents.

Provides tools for reading, writing, deleting, searching, and listing
entries in a LangGraph ``BaseStore``.  All store access goes through
:class:`AccessControlledStore` which handles namespace parsing and
access control.

Namespace format
~~~~~~~~~~~~~~~~
All namespace parameters use **dot-separated strings** that are converted
to tuples internally (e.g. ``"cache.tool_a"`` → ``("cache", "tool_a")``).
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime
from pydantic import BaseModel

from .store import AccessControlledStore

# ── Output models ────────────────────────────────────────────────────────────


class StoreEntry(BaseModel):
    """Output schema for store_get and store_search items."""

    namespace: str
    """Dot-separated namespace of the entry."""

    key: str
    """Entry key within the namespace."""

    value: dict[str, Any]
    """Entry value (JSON object)."""

    created_at: str | None = None
    """ISO-8601 creation timestamp, if available."""

    updated_at: str | None = None
    """ISO-8601 last-update timestamp, if available."""


# ── Get ──────────────────────────────────────────────────────────────────────


@tool(
    parse_docstring=True,
    extras={"output_schema": StoreEntry.model_json_schema()},
)
async def store_get(namespace: str, key: str, runtime: ToolRuntime) -> dict[str, Any]:
    """Read a single entry from the store.

    Args:
        namespace: Dot-separated namespace (e.g. ``"computed.dcf_model"``).
        key: Entry key within the namespace.
        runtime: Injected by LangGraph ToolNode.

    Returns:
        Dict with entry value and metadata.
    """
    store = AccessControlledStore.from_runtime(runtime)
    item = await store.aget(namespace, key)
    if item is None:
        raise ValueError(f"no entry found at namespace={namespace!r}, key={key!r}")
    return StoreEntry(
        namespace=namespace,
        key=item.key,
        value=item.value,
        created_at=item.created_at.isoformat() if item.created_at else None,
        updated_at=item.updated_at.isoformat() if item.updated_at else None,
    ).model_dump()


# ── Put ──────────────────────────────────────────────────────────────────────


@tool(parse_docstring=True)
async def store_put(
    namespace: str,
    key: str,
    value: str,
    runtime: ToolRuntime,
) -> str:
    """Write an entry to the store.

    Args:
        namespace: Dot-separated namespace (e.g. ``"computed.dcf_model"``).
        key: Entry key within the namespace.
        value: JSON string to store (must be a JSON object/dict).
        runtime: Injected by LangGraph ToolNode.

    Returns:
        Confirmation message.
    """
    store = AccessControlledStore.from_runtime(runtime)

    try:
        parsed = json.loads(value)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError(f"invalid JSON value — {exc}") from exc

    if not isinstance(parsed, dict):
        raise ValueError("value must be a JSON object (dict), not a scalar or array")

    await store.aput(namespace, key, parsed)
    return f"Stored at namespace={namespace!r}, key={key!r}"


# ── Delete ───────────────────────────────────────────────────────────────────


@tool(parse_docstring=True)
async def store_delete(namespace: str, key: str, runtime: ToolRuntime) -> str:
    """Delete an entry from the store.

    Args:
        namespace: Dot-separated namespace (e.g. ``"computed.dcf_model"``).
        key: Entry key within the namespace.
        runtime: Injected by LangGraph ToolNode.

    Returns:
        Confirmation message.
    """
    store = AccessControlledStore.from_runtime(runtime)
    await store.adelete(namespace, key)
    return f"Deleted key={key!r} from namespace={namespace!r}"


# ── Search ───────────────────────────────────────────────────────────────────


@tool(parse_docstring=True)
async def store_search(
    namespace: str,
    runtime: ToolRuntime,
    query: str | None = None,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Search entries within a namespace.

    Args:
        namespace: Dot-separated namespace prefix to search.
        query: Optional natural-language query for semantic search.
        limit: Maximum number of results (default 10).
        runtime: Injected by LangGraph ToolNode.

    Returns:
        List of matching entries.
    """
    store = AccessControlledStore.from_runtime(runtime)
    items = await store.asearch(namespace, query=query, limit=limit)
    return [
        StoreEntry(
            namespace=".".join(item.namespace),
            key=item.key,
            value=item.value,
            created_at=(item.created_at.isoformat() if item.created_at else None),
            updated_at=(item.updated_at.isoformat() if item.updated_at else None),
        ).model_dump()
        for item in items
    ]


# ── List Namespaces ──────────────────────────────────────────────────────────


@tool(parse_docstring=True)
async def store_list_namespaces(
    runtime: ToolRuntime,
    prefix: str | None = None,
) -> list[str]:
    """List namespaces in the store.

    Args:
        prefix: Optional dot-separated prefix to filter by.
        runtime: Injected by LangGraph ToolNode.

    Returns:
        List of dot-separated namespace strings.
    """
    store = AccessControlledStore.from_runtime(runtime)
    namespaces = await store.alist_namespaces(prefix=prefix)
    return [".".join(ns) for ns in namespaces]
