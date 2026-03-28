"""Tests for ToolResultCacheMiddleware."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import ToolMessage

from muffin_agent.middlewares.tool_result_cache import (
    ToolResultCacheMiddleware,
    get_args_hash,
    is_error_content,
)

# ── get_args_hash ────────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestArgsHash:
    """Test deterministic hash generation."""

    def test_deterministic_same_args(self):
        h1 = get_args_hash({"x": 1, "y": 2})
        h2 = get_args_hash({"x": 1, "y": 2})
        assert h1 == h2

    def test_arg_order_does_not_matter(self):
        h1 = get_args_hash({"y": 2, "x": 1})
        h2 = get_args_hash({"x": 1, "y": 2})
        assert h1 == h2

    def test_different_args_different_hash(self):
        h1 = get_args_hash({"x": 1})
        h2 = get_args_hash({"x": 2})
        assert h1 != h2

    def test_hash_length(self):
        h = get_args_hash({"symbol": "AAPL"})
        assert len(h) == 12

    def test_empty_args(self):
        h = get_args_hash({})
        assert len(h) == 12


# ── is_error_content ────────────────────────────────────────────────────────


@pytest.mark.unit
class TestIsErrorContent:
    """Test error content detection."""

    def test_error_prefix(self):
        assert is_error_content("Error: something broke") is True

    def test_error_permanent(self):
        assert is_error_content("Error (permanent): unauthorized") is True

    def test_duplicate_blocked(self):
        assert is_error_content("DUPLICATE CALL BLOCKED: ...") is True

    def test_normal_content(self):
        assert is_error_content('{"data": [1, 2, 3]}') is False

    def test_non_string(self):
        assert is_error_content(123) is False  # type: ignore[arg-type]


# ── ToolResultCacheMiddleware ────────────────────────────────────────────────


def _make_store_item(content: str, tool_name: str = "tool_a", args: dict | None = None):
    """Create a mock store Item."""
    item = MagicMock()
    item.value = {
        "content": content,
        "tool_name": tool_name,
        "args": args or {},
        "cached_at": "2026-03-22T14:30:00+00:00",
        "content_size": len(content),
    }
    item.key = get_args_hash(args or {})
    return item


def _make_request(
    tool_name: str,
    args: dict | None = None,
    tool_call_id: str = "tc_1",
    store: AsyncMock | None = None,
) -> MagicMock:
    """Create a mock ToolCallRequest with a store on runtime."""
    request = MagicMock()
    request.tool_call = {
        "name": tool_name,
        "args": args or {},
        "id": tool_call_id,
    }
    request.runtime = MagicMock()
    request.runtime.store = store
    return request


@pytest.mark.unit
class TestToolResultCacheMiddleware:
    """Test the cache middleware end-to-end."""

    def test_exposes_middleware_tools(self):
        """Middleware registers all three cache-related tools."""
        mw = ToolResultCacheMiddleware()
        assert len(mw.tools) == 3
        tool_names = {t.name for t in mw.tools}
        assert tool_names == {
            "discover_cached_tool_outputs",
            "get_tool_output_schema",
            "write_cached_tool_output_to_backend",
        }

    @pytest.mark.asyncio
    async def test_cache_miss_executes_handler_and_writes_to_store(self):
        """First call: handler executes, result cached to store, annotation appended."""
        store = AsyncMock()
        store.aget = AsyncMock(return_value=None)
        store.aput = AsyncMock()

        handler_result = ToolMessage(content="price data here", tool_call_id="tc_1")
        handler = AsyncMock(return_value=handler_result)

        mw = ToolResultCacheMiddleware()
        request = _make_request(
            "equity_price_historical",
            {"symbol": "AAPL"},
            store=store,
        )

        result = await mw.awrap_tool_call(request, handler)

        handler.assert_awaited_once_with(request)
        store.aput.assert_awaited_once()
        call_args = store.aput.call_args
        assert call_args.args[0] == ("cache", "equity_price_historical")
        assert call_args.args[1] == get_args_hash({"symbol": "AAPL"})
        assert call_args.args[2]["content"] == "price data here"
        assert call_args.args[2]["tool_name"] == "equity_price_historical"
        assert "[Data cached" in result.content
        assert "price data here" in result.content

    @pytest.mark.asyncio
    async def test_cache_hit_returns_cached_without_executing(self):
        """Duplicate call: returns cached content, handler NOT called."""
        store = AsyncMock()
        store.aget = AsyncMock(
            return_value=_make_store_item("cached price data"),
        )

        handler = AsyncMock()

        mw = ToolResultCacheMiddleware()
        request = _make_request(
            "equity_price_historical",
            {"symbol": "AAPL"},
            store=store,
        )

        result = await mw.awrap_tool_call(request, handler)

        handler.assert_not_awaited()
        assert "[Cached result" in result.content
        assert "cached price data" in result.content
        assert result.tool_call_id == "tc_1"

    @pytest.mark.asyncio
    async def test_error_results_not_cached(self):
        """Error tool messages are not written to the store."""
        store = AsyncMock()
        store.aget = AsyncMock(return_value=None)
        store.aput = AsyncMock()

        handler_result = ToolMessage(content="Error: unauthorized", tool_call_id="tc_1")
        handler = AsyncMock(return_value=handler_result)

        mw = ToolResultCacheMiddleware()
        request = _make_request("tool_a", store=store)

        result = await mw.awrap_tool_call(request, handler)

        store.aput.assert_not_awaited()
        assert result.content == "Error: unauthorized"

    @pytest.mark.asyncio
    async def test_cacheable_tools_whitelist_passes_through(self):
        """Non-whitelisted tools pass through uncached."""
        store = AsyncMock()
        handler_result = ToolMessage(content="result", tool_call_id="tc_1")
        handler = AsyncMock(return_value=handler_result)

        mw = ToolResultCacheMiddleware(cacheable_tools=frozenset({"tool_a"}))
        request = _make_request("tool_b", store=store)

        result = await mw.awrap_tool_call(request, handler)

        handler.assert_awaited_once()
        store.aget.assert_not_awaited()
        store.aput.assert_not_awaited()
        assert result.content == "result"

    @pytest.mark.asyncio
    async def test_cacheable_tools_whitelist_caches_listed_tool(self):
        """Whitelisted tool is cached normally."""
        store = AsyncMock()
        store.aget = AsyncMock(return_value=None)
        store.aput = AsyncMock()

        handler_result = ToolMessage(content="data", tool_call_id="tc_1")
        handler = AsyncMock(return_value=handler_result)

        mw = ToolResultCacheMiddleware(cacheable_tools=frozenset({"tool_a"}))
        request = _make_request("tool_a", store=store)

        result = await mw.awrap_tool_call(request, handler)

        handler.assert_awaited_once()
        store.aput.assert_awaited_once()
        assert "[Data cached" in result.content

    @pytest.mark.asyncio
    async def test_none_cacheable_tools_caches_all(self):
        """When cacheable_tools is None, all tools are cached."""
        store = AsyncMock()
        store.aget = AsyncMock(return_value=None)
        store.aput = AsyncMock()

        handler_result = ToolMessage(content="any data", tool_call_id="tc_1")
        handler = AsyncMock(return_value=handler_result)

        mw = ToolResultCacheMiddleware()
        request = _make_request("any_tool_name", store=store)

        result = await mw.awrap_tool_call(request, handler)

        store.aput.assert_awaited_once()
        assert "[Data cached" in result.content

    @pytest.mark.asyncio
    async def test_store_write_failure_returns_original(self):
        """If store write fails, return original result without annotation."""
        store = AsyncMock()
        store.aget = AsyncMock(return_value=None)
        store.aput = AsyncMock(side_effect=Exception("write failed"))

        handler_result = ToolMessage(content="data", tool_call_id="tc_1")
        handler = AsyncMock(return_value=handler_result)

        mw = ToolResultCacheMiddleware()
        request = _make_request("tool_a", store=store)

        result = await mw.awrap_tool_call(request, handler)

        assert result.content == "data"
        assert "[Data cached" not in result.content

    @pytest.mark.asyncio
    async def test_empty_store_read_treated_as_miss(self):
        """Empty content from store is treated as cache miss."""
        store = AsyncMock()
        store.aget = AsyncMock(return_value=_make_store_item(""))
        store.aput = AsyncMock()

        handler_result = ToolMessage(content="fresh data", tool_call_id="tc_1")
        handler = AsyncMock(return_value=handler_result)

        mw = ToolResultCacheMiddleware()
        request = _make_request("tool_a", store=store)

        result = await mw.awrap_tool_call(request, handler)

        handler.assert_awaited_once()
        assert "[Data cached" in result.content

    @pytest.mark.asyncio
    async def test_command_results_not_cached(self):
        """Command results from handler (e.g. ToolErrorHandler) pass through."""
        from langgraph.types import Command

        store = AsyncMock()
        store.aget = AsyncMock(return_value=None)

        cmd = Command(update={"messages": []})
        handler = AsyncMock(return_value=cmd)

        mw = ToolResultCacheMiddleware()
        request = _make_request("tool_a", store=store)

        result = await mw.awrap_tool_call(request, handler)

        store.aput.assert_not_awaited()
        assert result is cmd

    @pytest.mark.asyncio
    async def test_no_store_passes_through(self):
        """When runtime.store is None, tool executes without caching."""
        handler_result = ToolMessage(content="data", tool_call_id="tc_1")
        handler = AsyncMock(return_value=handler_result)

        mw = ToolResultCacheMiddleware()
        request = _make_request("tool_a", store=None)

        result = await mw.awrap_tool_call(request, handler)

        handler.assert_awaited_once()
        assert result.content == "data"

    @pytest.mark.asyncio
    async def test_store_metadata_fields(self):
        """Store value contains required metadata fields."""
        store = AsyncMock()
        store.aget = AsyncMock(return_value=None)
        store.aput = AsyncMock()

        handler_result = ToolMessage(content="some data", tool_call_id="tc_1")
        handler = AsyncMock(return_value=handler_result)

        mw = ToolResultCacheMiddleware()
        request = _make_request("tool_a", {"key": "val"}, store=store)

        await mw.awrap_tool_call(request, handler)

        store_value = store.aput.call_args.args[2]
        assert store_value["tool_name"] == "tool_a"
        assert store_value["args"] == {"key": "val"}
        assert "cached_at" in store_value
        assert store_value["content_size"] == len("some data")
        assert store_value["content"] == "some data"
