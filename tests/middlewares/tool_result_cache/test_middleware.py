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

    def test_exposes_no_middleware_tools(self):
        """Cache tools are registered by with_sandbox(), not the middleware itself."""
        mw = ToolResultCacheMiddleware()
        assert mw.tools == []

    @pytest.mark.asyncio
    async def test_cache_miss_executes_handler_and_writes_to_store(self):
        """First call: handler executes, result cached, content stays strict."""
        store = AsyncMock()
        store.aget = AsyncMock(return_value=None)
        store.aput = AsyncMock()

        original = '{"price": 123.45}'
        handler_result = ToolMessage(content=original, tool_call_id="tc_1")
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
        assert call_args.args[2]["content"] == original
        assert call_args.args[2]["tool_name"] == "equity_price_historical"
        # Strict-content invariant: returned content is byte-for-byte the original.
        assert result.content == original
        # Cache provenance lives in additional_kwargs.
        cache_meta = result.additional_kwargs["cache"]
        assert cache_meta["hit"] is False
        assert cache_meta["tool_name"] == "equity_price_historical"
        assert cache_meta["byte_size"] == len(original)

    @pytest.mark.asyncio
    async def test_cache_hit_returns_cached_strict_content(self):
        """Cache hit returns the cached payload byte-for-byte (strict content)."""
        cached = '{"data": [1, 2, 3]}'
        store = AsyncMock()
        store.aget = AsyncMock(return_value=_make_store_item(cached))

        handler = AsyncMock()

        mw = ToolResultCacheMiddleware()
        request = _make_request(
            "equity_price_historical",
            {"symbol": "AAPL"},
            store=store,
        )

        result = await mw.awrap_tool_call(request, handler)

        handler.assert_not_awaited()
        assert result.content == cached
        assert result.tool_call_id == "tc_1"
        assert result.additional_kwargs["cache"]["hit"] is True

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
        assert "cache" not in result.additional_kwargs

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

        original = "data"
        handler_result = ToolMessage(content=original, tool_call_id="tc_1")
        handler = AsyncMock(return_value=handler_result)

        mw = ToolResultCacheMiddleware(cacheable_tools=frozenset({"tool_a"}))
        request = _make_request("tool_a", store=store)

        result = await mw.awrap_tool_call(request, handler)

        handler.assert_awaited_once()
        store.aput.assert_awaited_once()
        assert result.content == original
        assert result.additional_kwargs["cache"]["hit"] is False

    @pytest.mark.asyncio
    async def test_none_cacheable_tools_caches_all(self):
        """When cacheable_tools is None, all tools are cached."""
        store = AsyncMock()
        store.aget = AsyncMock(return_value=None)
        store.aput = AsyncMock()

        original = "any data"
        handler_result = ToolMessage(content=original, tool_call_id="tc_1")
        handler = AsyncMock(return_value=handler_result)

        mw = ToolResultCacheMiddleware()
        request = _make_request("any_tool_name", store=store)

        result = await mw.awrap_tool_call(request, handler)

        store.aput.assert_awaited_once()
        assert result.content == original
        assert result.additional_kwargs["cache"]["hit"] is False

    @pytest.mark.asyncio
    async def test_store_write_failure_returns_original_unannotated(self):
        """If store write fails, return the original handler result untouched."""
        store = AsyncMock()
        store.aget = AsyncMock(return_value=None)
        store.aput = AsyncMock(side_effect=Exception("write failed"))

        handler_result = ToolMessage(content="data", tool_call_id="tc_1")
        handler = AsyncMock(return_value=handler_result)

        mw = ToolResultCacheMiddleware()
        request = _make_request("tool_a", store=store)

        result = await mw.awrap_tool_call(request, handler)

        # Original message passes through unchanged.
        assert result is handler_result
        assert "cache" not in result.additional_kwargs

    @pytest.mark.asyncio
    async def test_empty_store_read_treated_as_miss(self):
        """Empty content from store is treated as cache miss."""
        store = AsyncMock()
        store.aget = AsyncMock(return_value=_make_store_item(""))
        store.aput = AsyncMock()

        original = "fresh data"
        handler_result = ToolMessage(content=original, tool_call_id="tc_1")
        handler = AsyncMock(return_value=handler_result)

        mw = ToolResultCacheMiddleware()
        request = _make_request("tool_a", store=store)

        result = await mw.awrap_tool_call(request, handler)

        handler.assert_awaited_once()
        assert result.content == original
        assert result.additional_kwargs["cache"]["hit"] is False

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


@pytest.mark.unit
class TestStatusAuthoritative:
    """``status`` is the authoritative success/failure signal, not a content-string
    heuristic — a result must never be cached, nor have its status dropped on
    reconstruction, just because its content string doesn't start with "error".
    """

    @pytest.mark.asyncio
    async def test_status_error_not_cached_even_without_error_prefix(self):
        """Reproduces the ``get_indicators`` bug: ``ToolRetryMiddleware``'s
        exhausted-retry message doesn't start with "error" but carries
        ``status="error"`` — it must not be cached, and the returned message
        must keep ``status="error"``."""
        store = AsyncMock()
        store.aget = AsyncMock(return_value=None)
        store.aput = AsyncMock()

        content = "Tool 'get_indicators' failed after 1 attempt with ToolException: ..."
        handler_result = ToolMessage(
            content=content, tool_call_id="tc_1", status="error"
        )
        handler = AsyncMock(return_value=handler_result)

        mw = ToolResultCacheMiddleware()
        request = _make_request("get_indicators", store=store)

        result = await mw.awrap_tool_call(request, handler)

        store.aput.assert_not_awaited()
        assert result.status == "error"
        assert result.content == content

    @pytest.mark.asyncio
    async def test_success_status_forwarded_on_cache_write(self):
        """A newly-cached ``ToolMessage`` forwards its original ``status``."""
        store = AsyncMock()
        store.aget = AsyncMock(return_value=None)
        store.aput = AsyncMock()

        handler_result = ToolMessage(
            content="ok data", tool_call_id="tc_1", status="success"
        )
        handler = AsyncMock(return_value=handler_result)

        mw = ToolResultCacheMiddleware()
        request = _make_request("tool_a", store=store)

        result = await mw.awrap_tool_call(request, handler)

        store.aput.assert_awaited_once()
        assert result.status == "success"


@pytest.mark.unit
class TestStrictContentInvariant:
    """Verify the cache never injects prose / markers into ``content``.

    Size-based offload of oversized payloads is delegated to
    ``deepagents.middleware.filesystem.FilesystemMiddleware``, so this
    middleware's only job is to keep tool content byte-for-byte and stash
    cache provenance in ``additional_kwargs``.
    """

    @pytest.mark.asyncio
    async def test_large_payload_passes_through_unchanged(self):
        """A 50 KB JSON payload is cached and returned byte-for-byte."""
        store = AsyncMock()
        store.aget = AsyncMock(return_value=None)
        store.aput = AsyncMock()

        large = '{"rows":[' + ",".join(str(i) for i in range(10_000)) + "]}"
        handler = AsyncMock(return_value=ToolMessage(content=large, tool_call_id="tc"))
        mw = ToolResultCacheMiddleware()
        request = _make_request("tool_a", store=store)

        result = await mw.awrap_tool_call(request, handler)

        assert result.content == large
        cache_meta = result.additional_kwargs["cache"]
        assert cache_meta["hit"] is False
        assert cache_meta["byte_size"] == len(large)
        # Full payload landed in the store; no descriptor / preview shenanigans.
        assert store.aput.call_args.args[2]["content"] == large

    @pytest.mark.asyncio
    async def test_cache_hit_returns_byte_identical_payload(self):
        """Cache hits replay the cached content exactly, no prefix/suffix."""
        large = "y" * 50_000
        store = AsyncMock()
        store.aget = AsyncMock(return_value=_make_store_item(large))
        handler = AsyncMock()

        mw = ToolResultCacheMiddleware()
        request = _make_request("tool_a", store=store)

        result = await mw.awrap_tool_call(request, handler)

        handler.assert_not_awaited()
        assert result.content == large
        assert result.additional_kwargs["cache"]["hit"] is True
        assert result.additional_kwargs["cache"]["byte_size"] == len(large)
