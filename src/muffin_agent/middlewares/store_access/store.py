"""Access-controlled wrapper around LangGraph BaseStore."""

from __future__ import annotations

from typing import Any

from langgraph.prebuilt import ToolRuntime
from langgraph.store.base import BaseStore, Item, SearchItem

from .config import StoreConfiguration


def _parse_namespace(ns_string: str) -> tuple[str, ...]:
    """Convert a dot-separated namespace string to a tuple.

    Args:
        ns_string: Dot-separated namespace (e.g. ``"cache.tool_a"``).

    Returns:
        Tuple of namespace components (e.g. ``("cache", "tool_a")``).

    Raises:
        ValueError: If the namespace string is empty.
    """
    if not ns_string or not ns_string.strip():
        raise ValueError("Namespace string must not be empty")
    return tuple(ns_string.strip().split("."))


class AccessControlledStore:
    """Wrap a ``BaseStore`` with namespace parsing and access control.

    Accepts dot-separated namespace strings, converts to tuples, and
    validates against allowed namespace prefixes before every operation.
    """

    def __init__(
        self,
        store: BaseStore,
        allowed_namespaces: list[str] | None = None,
    ) -> None:
        """Initialize with a store and optional namespace restrictions."""
        self._store = store
        self._allowed = allowed_namespaces

    @classmethod
    def from_runtime(cls, runtime: ToolRuntime) -> AccessControlledStore:
        """Build from a ToolRuntime, raising if no store is available."""
        if runtime.store is None:
            raise ValueError("no store available")
        config = StoreConfiguration.from_runnable_config(runtime.config)
        return cls(runtime.store, config.store_allowed_namespaces)

    def _resolve(self, namespace: str) -> tuple[str, ...]:
        """Parse namespace string and check access."""
        ns = _parse_namespace(namespace)
        if self._allowed is not None and (not ns or ns[0] not in self._allowed):
            raise ValueError(
                f"Access denied: namespace {'.'.join(ns)!r} "
                f"not in allowed prefixes {self._allowed}"
            )
        return ns

    async def aget(self, namespace: str, key: str) -> Item | None:
        """Read a single entry from the store."""
        return await self._store.aget(self._resolve(namespace), key)

    async def aput(self, namespace: str, key: str, value: dict[str, Any]) -> None:
        """Write an entry to the store."""
        await self._store.aput(self._resolve(namespace), key, value)

    async def adelete(self, namespace: str, key: str) -> None:
        """Delete an entry from the store."""
        await self._store.adelete(self._resolve(namespace), key)

    async def asearch(
        self,
        namespace: str,
        *,
        query: str | None = None,
        limit: int = 10,
    ) -> list[SearchItem]:
        """Search entries within a namespace."""
        return await self._store.asearch(
            self._resolve(namespace), query=query, limit=limit
        )

    async def alist_namespaces(
        self,
        *,
        prefix: str | None = None,
    ) -> list[tuple[str, ...]]:
        """List namespaces, optionally filtered by prefix."""
        ns_prefix = self._resolve(prefix) if prefix else None
        return await self._store.alist_namespaces(prefix=ns_prefix)
