"""Tests for the shared cache primitives in ``cache.py``.

Covers:

* :func:`cache_lookup` — HIT/MISS behaviour, store errors, empty content.
* :func:`cache_store` — write success, skip patterns, error content, store
  errors.
* :func:`cached_invoke` — end-to-end HIT short-circuit, MISS execute+write,
  ``None`` store passthrough, ToolMessage content unwrapping.
* :func:`is_content_cacheable` — content-type filtering.
* **Parity** — cache keys / namespaces written by ``cache_store`` are read
  back correctly by ``cache_lookup``, and the middleware-driven path uses
  the same primitives so HITs cross between LLM-driven and direct callers.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import ToolMessage

from muffin_agent.middlewares.tool_result_cache import (
    cache_lookup,
    cache_store,
    cached_invoke,
    get_args_hash,
    is_content_cacheable,
)
from muffin_agent.middlewares.tool_result_cache.middleware import (
    ToolResultCacheMiddleware,
)

# ── is_content_cacheable ───────────────────────────────────────────────────────


@pytest.mark.unit
class TestIsContentCacheable:
    """Filter logic shared by cache_store and the middleware."""

    def test_normal_string_cacheable(self):
        assert is_content_cacheable('{"data": [1, 2, 3]}') is True

    def test_error_string_not_cacheable(self):
        assert is_content_cacheable("Error: something broke") is False

    def test_duplicate_blocked_string_not_cacheable(self):
        assert is_content_cacheable("DUPLICATE CALL BLOCKED: foo") is False

    def test_pattern_match_not_cacheable(self):
        assert (
            is_content_cacheable(
                "Tool result too large — offloaded to /tmp/x",
                ["tool result too large"],
            )
            is False
        )

    def test_pattern_match_case_insensitive(self):
        assert (
            is_content_cacheable("TOOL RESULT TOO LARGE", ["tool result too large"])
            is False
        )

    def test_list_content_cacheable(self):
        assert is_content_cacheable([{"x": 1}, {"y": 2}]) is True

    def test_list_content_ignores_string_patterns(self):
        # Skip patterns only apply to string content; list passes through.
        assert is_content_cacheable([{"x": 1}], ["error"]) is True

    def test_none_not_cacheable(self):
        assert is_content_cacheable(None) is False

    def test_dict_not_cacheable(self):
        assert is_content_cacheable({"x": 1}) is False


# ── cache_lookup ──────────────────────────────────────────────────────────────


def _make_store_item(content):
    """Create a mock store Item with cached *content*."""
    item = MagicMock()
    item.value = {"content": content}
    return item


@pytest.mark.unit
class TestCacheLookup:
    """HIT/MISS behaviour."""

    @pytest.mark.asyncio
    async def test_hit_returns_content(self):
        store = AsyncMock()
        store.aget = AsyncMock(return_value=_make_store_item('{"x": 1}'))
        hit, content = await cache_lookup("tool_a", {"symbol": "AAPL"}, store)
        assert hit is True
        assert content == '{"x": 1}'
        # Confirm exact namespace + key used (parity contract)
        store.aget.assert_awaited_once_with(
            ("cache", "tool_a"), get_args_hash({"symbol": "AAPL"})
        )

    @pytest.mark.asyncio
    async def test_miss_returns_none(self):
        store = AsyncMock()
        store.aget = AsyncMock(return_value=None)
        hit, content = await cache_lookup("tool_a", {"symbol": "AAPL"}, store)
        assert hit is False
        assert content is None

    @pytest.mark.asyncio
    async def test_empty_content_treated_as_miss(self):
        store = AsyncMock()
        store.aget = AsyncMock(return_value=_make_store_item(""))
        hit, content = await cache_lookup("tool_a", {"symbol": "AAPL"}, store)
        assert hit is False
        assert content is None

    @pytest.mark.asyncio
    async def test_store_error_treated_as_miss(self):
        store = AsyncMock()
        store.aget = AsyncMock(side_effect=RuntimeError("store down"))
        hit, content = await cache_lookup("tool_a", {"symbol": "AAPL"}, store)
        assert hit is False
        assert content is None

    @pytest.mark.asyncio
    async def test_list_content_hit(self):
        store = AsyncMock()
        store.aget = AsyncMock(return_value=_make_store_item([{"x": 1}]))
        hit, content = await cache_lookup("tool_a", {}, store)
        assert hit is True
        assert content == [{"x": 1}]


# ── cache_store ───────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestCacheStore:
    """Write path."""

    @pytest.mark.asyncio
    async def test_string_content_written(self):
        store = AsyncMock()
        store.aput = AsyncMock()
        written = await cache_store("tool_a", {"sym": "AAPL"}, store, '{"x": 1}')
        assert written is True
        call = store.aput.await_args
        assert call.args[0] == ("cache", "tool_a")
        assert call.args[1] == get_args_hash({"sym": "AAPL"})
        payload = call.args[2]
        assert payload["content"] == '{"x": 1}'
        assert payload["tool_name"] == "tool_a"
        assert payload["args"] == {"sym": "AAPL"}
        assert "cached_at" in payload
        assert payload["content_size"] == len('{"x": 1}')

    @pytest.mark.asyncio
    async def test_list_content_written(self):
        store = AsyncMock()
        store.aput = AsyncMock()
        written = await cache_store("tool_a", {}, store, [{"x": 1}])
        assert written is True

    @pytest.mark.asyncio
    async def test_error_content_skipped(self):
        store = AsyncMock()
        store.aput = AsyncMock()
        written = await cache_store("tool_a", {}, store, "Error: nope")
        assert written is False
        store.aput.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_pattern_match_skipped(self):
        store = AsyncMock()
        store.aput = AsyncMock()
        written = await cache_store(
            "tool_a",
            {},
            store,
            "Tool result too large — offloaded",
            non_cacheable_patterns=["tool result too large"],
        )
        assert written is False
        store.aput.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_non_cacheable_type_skipped(self):
        store = AsyncMock()
        store.aput = AsyncMock()
        written = await cache_store("tool_a", {}, store, None)
        assert written is False
        store.aput.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_store_error_returns_false(self):
        store = AsyncMock()
        store.aput = AsyncMock(side_effect=RuntimeError("store down"))
        written = await cache_store("tool_a", {}, store, '{"x": 1}')
        assert written is False


# ── cached_invoke ─────────────────────────────────────────────────────────────


def _make_tool(name: str, return_value):
    """Build a mock BaseTool with ``.name`` and ``.ainvoke``."""
    tool = MagicMock()
    tool.name = name
    tool.ainvoke = AsyncMock(return_value=return_value)
    return tool


@pytest.mark.unit
class TestCachedInvoke:
    """High-level wrapper for non-LLM callers."""

    @pytest.mark.asyncio
    async def test_cache_hit_short_circuits(self):
        store = AsyncMock()
        store.aget = AsyncMock(return_value=_make_store_item('{"cached": true}'))
        store.aput = AsyncMock()
        tool = _make_tool("tool_a", "should not be called")
        result = await cached_invoke(tool, {"x": 1}, store)
        assert result == '{"cached": true}'
        tool.ainvoke.assert_not_awaited()
        store.aput.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_cache_miss_executes_and_writes(self):
        store = AsyncMock()
        store.aget = AsyncMock(return_value=None)
        store.aput = AsyncMock()
        tool = _make_tool(
            "tool_a",
            ToolMessage(content='{"fresh": true}', tool_call_id="tc_1"),
        )
        result = await cached_invoke(tool, {"x": 1}, store)
        assert result == '{"fresh": true}'
        tool.ainvoke.assert_awaited_once_with({"x": 1})
        store.aput.assert_awaited_once()
        # Namespace + key match the lookup path
        ns, key, payload = store.aput.await_args.args
        assert ns == ("cache", "tool_a")
        assert key == get_args_hash({"x": 1})
        assert payload["content"] == '{"fresh": true}'

    @pytest.mark.asyncio
    async def test_unwraps_tool_message_content(self):
        store = AsyncMock()
        store.aget = AsyncMock(return_value=None)
        store.aput = AsyncMock()
        tool = _make_tool(
            "tool_a",
            ToolMessage(content=[{"row": 1}], tool_call_id="tc_1"),
        )
        result = await cached_invoke(tool, {}, store)
        assert result == [{"row": 1}]
        # Stored content is the unwrapped list, not the ToolMessage
        stored_content = store.aput.await_args.args[2]["content"]
        assert stored_content == [{"row": 1}]

    @pytest.mark.asyncio
    async def test_returns_raw_when_tool_does_not_wrap_in_tool_message(self):
        store = AsyncMock()
        store.aget = AsyncMock(return_value=None)
        store.aput = AsyncMock()
        tool = _make_tool("tool_a", '{"raw": true}')
        result = await cached_invoke(tool, {}, store)
        assert result == '{"raw": true}'
        store.aput.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_none_store_bypasses_cache(self):
        tool = _make_tool(
            "tool_a", ToolMessage(content='{"raw": true}', tool_call_id="tc_1")
        )
        result = await cached_invoke(tool, {}, None)
        assert result == '{"raw": true}'
        tool.ainvoke.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_tool_name_override(self):
        store = AsyncMock()
        store.aget = AsyncMock(return_value=None)
        store.aput = AsyncMock()
        tool = _make_tool("real_name", "ok")
        await cached_invoke(tool, {}, store, tool_name="aliased")
        # Lookup used the override
        store.aget.assert_awaited_once()
        ns, _ = store.aget.await_args.args
        assert ns == ("cache", "aliased")
        # Write used the override
        store.aput.assert_awaited_once()
        ns_put, _, payload = store.aput.await_args.args
        assert ns_put == ("cache", "aliased")
        assert payload["tool_name"] == "aliased"

    @pytest.mark.asyncio
    async def test_non_cacheable_pattern_propagates(self):
        store = AsyncMock()
        store.aget = AsyncMock(return_value=None)
        store.aput = AsyncMock()
        tool = _make_tool(
            "tool_a",
            ToolMessage(
                content="Tool result too large — offloaded to /tmp/x",
                tool_call_id="tc_1",
            ),
        )
        result = await cached_invoke(
            tool, {}, store, non_cacheable_patterns=["tool result too large"]
        )
        # Result still returned to caller
        assert "too large" in result
        # But NOT cached
        store.aput.assert_not_awaited()


# ── Parity between cached_invoke and the middleware ───────────────────────────


@pytest.mark.unit
class TestCachedInvokeMiddlewareParity:
    """Cache keys + payloads must collide perfectly across both paths."""

    @pytest.mark.asyncio
    async def test_cached_invoke_writes_a_hit_for_middleware(self):
        """Direct write → middleware read sees a HIT."""
        from langchain.tools.tool_node import ToolCallRequest  # noqa: F401

        # Shared store mock that records writes and serves them back on read
        cache: dict[tuple, dict] = {}

        async def aget(namespace, key):
            entry = cache.get((namespace, key))
            if entry is None:
                return None
            item = MagicMock()
            item.value = entry
            return item

        async def aput(namespace, key, value):
            cache[(namespace, key)] = value

        store = MagicMock()
        store.aget = aget
        store.aput = aput

        # Step 1: direct invocation writes via cached_invoke.
        tool = _make_tool(
            "tool_a",
            ToolMessage(content='{"direct": true}', tool_call_id="tc_direct"),
        )
        await cached_invoke(tool, {"sym": "AAPL"}, store)

        # Step 2: middleware reads via awrap_tool_call → expect HIT.
        middleware = ToolResultCacheMiddleware()
        request = MagicMock()
        request.tool_call = {
            "name": "tool_a",
            "args": {"sym": "AAPL"},
            "id": "tc_middleware",
        }
        request.runtime = MagicMock()
        request.runtime.store = store
        request.runtime.config = {}

        handler = AsyncMock()  # should NOT be called on HIT
        result = await middleware.awrap_tool_call(request, handler)

        assert isinstance(result, ToolMessage)
        assert result.content == '{"direct": true}'
        assert result.additional_kwargs["cache"]["hit"] is True
        assert result.additional_kwargs["cache"]["tool_name"] == "tool_a"
        handler.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_middleware_write_serves_cached_invoke_hit(self):
        """Middleware write → cached_invoke read sees a HIT."""
        cache: dict[tuple, dict] = {}

        async def aget(namespace, key):
            entry = cache.get((namespace, key))
            if entry is None:
                return None
            item = MagicMock()
            item.value = entry
            return item

        async def aput(namespace, key, value):
            cache[(namespace, key)] = value

        store = MagicMock()
        store.aget = aget
        store.aput = aput

        # Step 1: middleware execution writes via cache_store under the hood.
        middleware = ToolResultCacheMiddleware()
        request = MagicMock()
        request.tool_call = {
            "name": "tool_a",
            "args": {"sym": "AAPL"},
            "id": "tc_mid",
        }
        request.runtime = MagicMock()
        request.runtime.store = store
        request.runtime.config = {}

        async def handler(req):
            return ToolMessage(content='{"middleware": true}', tool_call_id="tc_mid")

        await middleware.awrap_tool_call(request, handler)

        # Step 2: direct cached_invoke reads → HIT (without re-invoking the tool).
        tool = _make_tool("tool_a", "should NOT be called")
        result = await cached_invoke(tool, {"sym": "AAPL"}, store)
        assert result == '{"middleware": true}'
        tool.ainvoke.assert_not_awaited()
