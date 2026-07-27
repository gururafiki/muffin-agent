"""The MCP tool-discovery cache in ``data_collection.utils``.

This cache is the fix for the API's read latency. LangGraph Platform rebuilds a
factory-registered graph on every request — including plain reads like
``POST /threads/{id}/history`` — and each agent factory in that graph calls
``get_tools``. Uncached, building ``criteria_analysis`` opened 23 MCP sessions at
~1.1 s each, which accounted for that graph's entire 27.3 s history read.

The regression these tests guard is subtle: a cache that is per-*call* rather
than per-connection-set, or one that swallows failures permanently, would still
pass every other test in the suite while quietly restoring the latency.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from muffin_agent.agents.data_collection.utils import get_tools, reset_mcp_tool_cache

_CFG: dict[str, Any] = {"configurable": {}}
_OTHER_CFG: dict[str, Any] = {
    "configurable": {"openbb_mcp_url": "http://elsewhere:9999/mcp"}
}


def _tool(name: str) -> MagicMock:
    tool = MagicMock()
    tool.name = name
    return tool


def _client_returning(*names: str) -> tuple[MagicMock, AsyncMock]:
    """A stand-in ``MultiServerMCPClient`` class plus its ``get_tools`` mock."""
    get = AsyncMock(return_value=[_tool(n) for n in names])
    instance = MagicMock()
    instance.get_tools = get
    return MagicMock(return_value=instance), get


@pytest.fixture(autouse=True)
def _clean_cache():
    reset_mcp_tool_cache()
    yield
    reset_mcp_tool_cache()


@pytest.mark.unit
@pytest.mark.asyncio
class TestMcpToolCache:
    async def test_repeated_loads_hit_the_server_once(self):
        """The whole point: N agent factories, one round trip."""
        client_cls, get = _client_returning("get_price", "get_news")
        with patch(
            "muffin_agent.agents.data_collection.utils.MultiServerMCPClient", client_cls
        ):
            for _ in range(23):  # a criteria_analysis-sized graph build
                loaded = await get_tools(_CFG, ["get_price"])
                assert [t.name for t in loaded] == ["get_price"]
        assert get.await_count == 1

    async def test_concurrent_loads_hit_the_server_once(self):
        """A build that fans out must not stampede past the empty cache."""
        client_cls, get = _client_returning("get_price")
        with patch(
            "muffin_agent.agents.data_collection.utils.MultiServerMCPClient", client_cls
        ):
            await asyncio.gather(*(get_tools(_CFG, ["get_price"]) for _ in range(10)))
        assert get.await_count == 1

    async def test_each_filter_still_gets_only_its_own_tools(self):
        """Caching the full list must not leak tools between callers."""
        client_cls, _ = _client_returning("get_price", "get_news", "scrape")
        with patch(
            "muffin_agent.agents.data_collection.utils.MultiServerMCPClient", client_cls
        ):
            first = await get_tools(_CFG, ["get_price"])
            second = await get_tools(_CFG, ["scrape", "get_news"])
        assert [t.name for t in first] == ["get_price"]
        assert sorted(t.name for t in second) == ["get_news", "scrape"]

    async def test_different_mcp_urls_do_not_share_a_cache_entry(self):
        """Local-dev and deployed configs point at different servers."""
        client_cls, get = _client_returning("get_price")
        with patch(
            "muffin_agent.agents.data_collection.utils.MultiServerMCPClient", client_cls
        ):
            await get_tools(_CFG, ["get_price"])
            await get_tools(_OTHER_CFG, ["get_price"])
        assert get.await_count == 2

    async def test_failures_are_not_cached(self):
        """A transient MCP outage must not disable tools until the next deploy."""
        instance = MagicMock()
        instance.get_tools = AsyncMock(
            side_effect=[ConnectionError("mcp down"), [_tool("get_price")]]
        )
        with patch(
            "muffin_agent.agents.data_collection.utils.MultiServerMCPClient",
            MagicMock(return_value=instance),
        ):
            with pytest.raises(ConnectionError):
                await get_tools(_CFG, ["get_price"])
            retried = await get_tools(_CFG, ["get_price"])
        assert [t.name for t in retried] == ["get_price"]

    async def test_empty_allowlist_never_touches_the_network(self):
        client_cls, get = _client_returning("get_price")
        with patch(
            "muffin_agent.agents.data_collection.utils.MultiServerMCPClient", client_cls
        ):
            assert await get_tools(_CFG, [], custom_tools=[_tool("web_search")]) != []
        get.assert_not_awaited()

    async def test_reset_forces_a_reload(self):
        """The hook the root conftest uses to isolate tests from each other."""
        client_cls, get = _client_returning("get_price")
        with patch(
            "muffin_agent.agents.data_collection.utils.MultiServerMCPClient", client_cls
        ):
            await get_tools(_CFG, ["get_price"])
            reset_mcp_tool_cache()
            await get_tools(_CFG, ["get_price"])
        assert get.await_count == 2
