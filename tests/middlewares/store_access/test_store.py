"""Unit tests for AccessControlledStore."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from muffin_agent.middlewares.store_access.store import (
    AccessControlledStore,
    _parse_namespace,
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


# ── _parse_namespace ─────────────────────────────────────────────────────────


@pytest.mark.unit
class TestParseNamespace:
    def test_single_component(self):
        assert _parse_namespace("cache") == ("cache",)

    def test_multi_component(self):
        assert _parse_namespace("cache.tool_a") == ("cache", "tool_a")

    def test_three_components(self):
        assert _parse_namespace("a.b.c") == ("a", "b", "c")

    def test_strips_whitespace(self):
        assert _parse_namespace("  cache.tool_a  ") == ("cache", "tool_a")

    def test_empty_string_raises(self):
        with pytest.raises(ValueError, match="must not be empty"):
            _parse_namespace("")

    def test_whitespace_only_raises(self):
        with pytest.raises(ValueError, match="must not be empty"):
            _parse_namespace("   ")


# ── from_runtime ─────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestFromRuntime:
    def test_no_store_raises(self):
        runtime = _make_runtime(store=None)
        with pytest.raises(ValueError, match="no store available"):
            AccessControlledStore.from_runtime(runtime)

    def test_builds_with_config(self):
        store = AsyncMock()
        runtime = _make_runtime(store=store, allowed_namespaces=["cache"])
        acs = AccessControlledStore.from_runtime(runtime)
        assert acs._allowed == ["cache"]

    def test_unrestricted_when_no_config(self):
        store = AsyncMock()
        runtime = _make_runtime(store=store)
        acs = AccessControlledStore.from_runtime(runtime)
        assert acs._allowed is None


# ── _resolve (access control) ───────────────────────────────────────────────


@pytest.mark.unit
class TestResolve:
    def test_valid_namespace(self):
        acs = AccessControlledStore(AsyncMock(), allowed_namespaces=["cache"])
        assert acs._resolve("cache.tool_a") == ("cache", "tool_a")

    def test_denied_namespace(self):
        acs = AccessControlledStore(AsyncMock(), allowed_namespaces=["cache"])
        with pytest.raises(ValueError, match="Access denied"):
            acs._resolve("secret.data")

    def test_unrestricted(self):
        acs = AccessControlledStore(AsyncMock(), allowed_namespaces=None)
        assert acs._resolve("anything.goes") == ("anything", "goes")

    def test_empty_string_raises(self):
        acs = AccessControlledStore(AsyncMock())
        with pytest.raises(ValueError, match="must not be empty"):
            acs._resolve("")

    def test_empty_allowed_list_denies(self):
        acs = AccessControlledStore(AsyncMock(), allowed_namespaces=[])
        with pytest.raises(ValueError, match="Access denied"):
            acs._resolve("cache")


# ── Delegation methods ───────────────────────────────────────────────────────


@pytest.mark.unit
class TestDelegation:
    @pytest.mark.asyncio
    async def test_aget_delegates(self):
        store = AsyncMock()
        store.aget = AsyncMock(return_value=None)
        acs = AccessControlledStore(store)

        await acs.aget("cache.tool_a", "key1")
        store.aget.assert_awaited_once_with(("cache", "tool_a"), "key1")

    @pytest.mark.asyncio
    async def test_aput_delegates(self):
        store = AsyncMock()
        store.aput = AsyncMock()
        acs = AccessControlledStore(store)

        await acs.aput("computed.dcf", "v1", {"nav": 150.5})
        store.aput.assert_awaited_once_with(("computed", "dcf"), "v1", {"nav": 150.5})

    @pytest.mark.asyncio
    async def test_adelete_delegates(self):
        store = AsyncMock()
        store.adelete = AsyncMock()
        acs = AccessControlledStore(store)

        await acs.adelete("cache.tool_a", "key1")
        store.adelete.assert_awaited_once_with(("cache", "tool_a"), "key1")

    @pytest.mark.asyncio
    async def test_asearch_delegates(self):
        store = AsyncMock()
        store.asearch = AsyncMock(return_value=[])
        acs = AccessControlledStore(store)

        await acs.asearch("cache", query="test", limit=5)
        store.asearch.assert_awaited_once_with(("cache",), query="test", limit=5)

    @pytest.mark.asyncio
    async def test_alist_namespaces_with_prefix(self):
        store = AsyncMock()
        store.alist_namespaces = AsyncMock(return_value=[])
        acs = AccessControlledStore(store)

        await acs.alist_namespaces(prefix="cache")
        store.alist_namespaces.assert_awaited_once_with(prefix=("cache",))

    @pytest.mark.asyncio
    async def test_alist_namespaces_no_prefix(self):
        store = AsyncMock()
        store.alist_namespaces = AsyncMock(return_value=[])
        acs = AccessControlledStore(store)

        await acs.alist_namespaces()
        store.alist_namespaces.assert_awaited_once_with(prefix=None)

    @pytest.mark.asyncio
    async def test_aget_denied(self):
        store = AsyncMock()
        acs = AccessControlledStore(store, allowed_namespaces=["cache"])

        with pytest.raises(ValueError, match="Access denied"):
            await acs.aget("secret.data", "key")
        store.aget.assert_not_awaited()
