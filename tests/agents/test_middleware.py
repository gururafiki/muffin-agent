"""Tests for tool result cache middleware."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import ToolMessage
from langgraph.types import Command

from muffin_agent.agents.middleware import (
    ToolResultCacheMiddleware,
    cache_key,
)


@pytest.mark.unit
class TestCacheKey:
    """Verify cache key generation."""

    def test_deterministic(self):
        tc = {"name": "compute_roic", "args": {"nopat": 100, "invested": 500}}
        assert cache_key(tc) == cache_key(tc)

    def test_sorted_args(self):
        tc1 = {"name": "f", "args": {"b": 2, "a": 1}}
        tc2 = {"name": "f", "args": {"a": 1, "b": 2}}
        assert cache_key(tc1) == cache_key(tc2)

    def test_different_tools(self):
        tc1 = {"name": "f", "args": {"a": 1}}
        tc2 = {"name": "g", "args": {"a": 1}}
        assert cache_key(tc1) != cache_key(tc2)


@pytest.mark.unit
class TestToolResultCacheMiddleware:
    """Verify cache hit, miss, and filtering behaviour."""

    def _make_request(self, tool_name, args, state=None):
        req = MagicMock()
        req.tool_call = {
            "name": tool_name,
            "args": args,
            "id": "call_123",
        }
        req.state = state or {}
        return req

    @pytest.mark.asyncio
    async def test_cache_miss_calls_handler(self):
        middleware = ToolResultCacheMiddleware()
        handler = AsyncMock(
            return_value=ToolMessage(content="result", tool_call_id="call_123")
        )
        req = self._make_request("compute_roic", {"nopat": 100})

        result = await middleware.awrap_tool_call(req, handler)

        handler.assert_awaited_once_with(req)
        # Returns Command to update cache state
        assert isinstance(result, Command)
        assert "cached_tool_results" in result.update
        assert "messages" in result.update

    @pytest.mark.asyncio
    async def test_cache_hit_skips_handler(self):
        middleware = ToolResultCacheMiddleware()
        key = cache_key({"name": "compute_roic", "args": {"nopat": 100}})
        req = self._make_request(
            "compute_roic",
            {"nopat": 100},
            state={"cached_tool_results": {key: "cached_result"}},
        )
        handler = AsyncMock()

        result = await middleware.awrap_tool_call(req, handler)

        handler.assert_not_awaited()
        assert isinstance(result, ToolMessage)
        assert "[cached]" in result.content
        assert "cached_result" in result.content

    @pytest.mark.asyncio
    async def test_non_cacheable_tool_bypasses_cache(self):
        middleware = ToolResultCacheMiddleware(
            cacheable_tools=frozenset({"compute_roic"})
        )
        handler = AsyncMock(
            return_value=ToolMessage(content="result", tool_call_id="call_123")
        )
        req = self._make_request("some_other_tool", {"x": 1})

        result = await middleware.awrap_tool_call(req, handler)

        handler.assert_awaited_once()
        # Returns plain ToolMessage (no caching)
        assert isinstance(result, ToolMessage)
        assert result.content == "result"

    @pytest.mark.asyncio
    async def test_error_result_not_cached(self):
        middleware = ToolResultCacheMiddleware()
        handler = AsyncMock(
            return_value=ToolMessage(
                content="Error: connection failed", tool_call_id="call_123"
            )
        )
        req = self._make_request("compute_roic", {"nopat": 100})

        result = await middleware.awrap_tool_call(req, handler)

        # Error results pass through without caching
        assert isinstance(result, ToolMessage)
        assert result.content == "Error: connection failed"

    @pytest.mark.asyncio
    async def test_cache_all_when_no_filter(self):
        middleware = ToolResultCacheMiddleware()
        handler = AsyncMock(
            return_value=ToolMessage(content="data", tool_call_id="call_123")
        )
        req = self._make_request("any_mcp_tool", {"ticker": "AAPL"})

        result = await middleware.awrap_tool_call(req, handler)

        assert isinstance(result, Command)
        assert "cached_tool_results" in result.update

    def test_immutable_cacheable_tools(self):
        tools = frozenset({"a", "b"})
        middleware = ToolResultCacheMiddleware(cacheable_tools=tools)
        assert middleware.cacheable_tools is tools
