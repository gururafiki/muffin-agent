"""Unit tests for store access tools."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from muffin_agent.middlewares.store_access.tools import (
    store_delete,
    store_get,
    store_list_namespaces,
    store_put,
    store_search,
)


def _make_runtime(store=None, allowed_namespaces=None):
    """Return a mock ToolRuntime."""
    runtime = MagicMock()
    configurable = {}
    if allowed_namespaces is not None:
        configurable["store_allowed_namespaces"] = allowed_namespaces
    runtime.config = {"configurable": configurable}
    runtime.store = store
    return runtime


# ── store_get ────────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestStoreGet:
    @pytest.mark.asyncio
    async def test_returns_value(self):
        item = MagicMock()
        item.key = "abc"
        item.value = {"data": 42}
        item.created_at = None
        item.updated_at = None

        store = AsyncMock()
        store.aget = AsyncMock(return_value=item)
        runtime = _make_runtime(store=store)

        result = await store_get.coroutine("cache.tool_a", "abc", runtime)
        assert result["value"] == {"data": 42}
        assert result["namespace"] == "cache.tool_a"
        assert result["key"] == "abc"

    @pytest.mark.asyncio
    async def test_not_found_raises(self):
        store = AsyncMock()
        store.aget = AsyncMock(return_value=None)
        runtime = _make_runtime(store=store)

        with pytest.raises(ValueError, match="no entry found"):
            await store_get.coroutine("cache.tool_a", "missing", runtime)

    @pytest.mark.asyncio
    async def test_no_store_raises(self):
        runtime = _make_runtime(store=None)
        with pytest.raises(ValueError, match="no store available"):
            await store_get.coroutine("cache.tool_a", "abc", runtime)

    @pytest.mark.asyncio
    async def test_namespace_denied_raises(self):
        store = AsyncMock()
        runtime = _make_runtime(store=store, allowed_namespaces=["cache"])
        with pytest.raises(ValueError, match="Access denied"):
            await store_get.coroutine("secret.data", "key", runtime)
        store.aget.assert_not_awaited()


# ── store_put ────────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestStorePut:
    @pytest.mark.asyncio
    async def test_stores_value(self):
        store = AsyncMock()
        store.aput = AsyncMock()
        runtime = _make_runtime(store=store)

        result = await store_put.coroutine(
            "computed.dcf", "model_v1", '{"nav": 150.5}', runtime
        )
        assert "Stored at" in result
        store.aput.assert_awaited_once_with(
            ("computed", "dcf"), "model_v1", {"nav": 150.5}
        )

    @pytest.mark.asyncio
    async def test_invalid_json_raises(self):
        store = AsyncMock()
        runtime = _make_runtime(store=store)
        with pytest.raises(ValueError, match="invalid JSON"):
            await store_put.coroutine("computed.dcf", "key", "not json", runtime)

    @pytest.mark.asyncio
    async def test_non_dict_json_raises(self):
        store = AsyncMock()
        runtime = _make_runtime(store=store)
        with pytest.raises(ValueError, match="JSON object"):
            await store_put.coroutine("computed.dcf", "key", "[1, 2]", runtime)

    @pytest.mark.asyncio
    async def test_no_store_raises(self):
        runtime = _make_runtime(store=None)
        with pytest.raises(ValueError, match="no store available"):
            await store_put.coroutine("computed.dcf", "key", '{"a": 1}', runtime)

    @pytest.mark.asyncio
    async def test_namespace_denied_raises(self):
        store = AsyncMock()
        runtime = _make_runtime(store=store, allowed_namespaces=["cache"])
        with pytest.raises(ValueError, match="Access denied"):
            await store_put.coroutine("secret.data", "key", '{"a": 1}', runtime)


# ── store_delete ─────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestStoreDelete:
    @pytest.mark.asyncio
    async def test_deletes_entry(self):
        store = AsyncMock()
        store.adelete = AsyncMock()
        runtime = _make_runtime(store=store)

        result = await store_delete.coroutine("cache.tool_a", "abc", runtime)
        assert "Deleted" in result
        store.adelete.assert_awaited_once_with(("cache", "tool_a"), "abc")

    @pytest.mark.asyncio
    async def test_no_store_raises(self):
        runtime = _make_runtime(store=None)
        with pytest.raises(ValueError, match="no store available"):
            await store_delete.coroutine("cache.tool_a", "abc", runtime)

    @pytest.mark.asyncio
    async def test_namespace_denied_raises(self):
        store = AsyncMock()
        runtime = _make_runtime(store=store, allowed_namespaces=["cache"])
        with pytest.raises(ValueError, match="Access denied"):
            await store_delete.coroutine("secret.data", "key", runtime)


# ── store_search ─────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestStoreSearch:
    @pytest.mark.asyncio
    async def test_returns_results(self):
        item = MagicMock()
        item.namespace = ("cache", "tool_a")
        item.key = "abc"
        item.value = {"data": 42}
        item.created_at = None
        item.updated_at = None

        store = AsyncMock()
        store.asearch = AsyncMock(return_value=[item])
        runtime = _make_runtime(store=store)

        result = await store_search.coroutine("cache.tool_a", runtime)
        assert len(result) == 1
        assert result[0]["key"] == "abc"

    @pytest.mark.asyncio
    async def test_empty_results(self):
        store = AsyncMock()
        store.asearch = AsyncMock(return_value=[])
        runtime = _make_runtime(store=store)

        result = await store_search.coroutine("cache", runtime)
        assert result == []

    @pytest.mark.asyncio
    async def test_no_store_raises(self):
        runtime = _make_runtime(store=None)
        with pytest.raises(ValueError, match="no store available"):
            await store_search.coroutine("cache", runtime)


# ── store_list_namespaces ────────────────────────────────────────────────────


@pytest.mark.unit
class TestStoreListNamespaces:
    @pytest.mark.asyncio
    async def test_returns_namespaces(self):
        store = AsyncMock()
        store.alist_namespaces = AsyncMock(
            return_value=[("cache", "tool_a"), ("cache", "tool_b")]
        )
        runtime = _make_runtime(store=store)

        result = await store_list_namespaces.coroutine(runtime, prefix="cache")
        assert result == ["cache.tool_a", "cache.tool_b"]

    @pytest.mark.asyncio
    async def test_no_prefix(self):
        store = AsyncMock()
        store.alist_namespaces = AsyncMock(return_value=[("cache",), ("computed",)])
        runtime = _make_runtime(store=store)

        result = await store_list_namespaces.coroutine(runtime)
        assert result == ["cache", "computed"]

    @pytest.mark.asyncio
    async def test_no_store_raises(self):
        runtime = _make_runtime(store=None)
        with pytest.raises(ValueError, match="no store available"):
            await store_list_namespaces.coroutine(runtime)

    @pytest.mark.asyncio
    async def test_namespace_denied_raises(self):
        store = AsyncMock()
        runtime = _make_runtime(store=store, allowed_namespaces=["cache"])
        with pytest.raises(ValueError, match="Access denied"):
            await store_list_namespaces.coroutine(runtime, prefix="secret")
