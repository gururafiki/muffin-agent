"""Shared utilities for data collection agents."""

from __future__ import annotations

import asyncio
import json
import time

from langchain_core.runnables import RunnableConfig
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.sessions import Connection

from ...mcp_config import McpConfiguration

# How long a discovered tool list stays usable. The MCP servers' tool lists only
# change when those services are redeployed, so this is a safety valve rather
# than a correctness mechanism — it bounds how long a rebuilt openbb-mcp can go
# unnoticed by an already-running api process.
_TOOL_CACHE_TTL_SECONDS = 900.0

# Keyed by the connection set, so a config pointing at different MCP URLs (local
# dev vs deployed) never reads another config's tools.
_tool_cache: dict[str, tuple[float, list]] = {}
_tool_cache_locks: dict[str, asyncio.Lock] = {}


def reset_mcp_tool_cache() -> None:
    """Drop every cached tool list. For tests, which patch the MCP client."""
    _tool_cache.clear()
    _tool_cache_locks.clear()


async def _load_all_mcp_tools(connections: dict[str, Connection]) -> list:
    """Every tool the configured MCP servers expose, cached per connection set.

    **Why this cache exists.** LangGraph Platform rebuilds a factory-registered
    graph (`make_graph`) on *every* API request — runs, but also plain reads like
    ``POST /threads/{id}/history``. Each agent factory in a graph calls
    :func:`get_tools`, and each call used to open a fresh MCP session just to list
    tools. Measured on the deployed node: ~1.1 s per round trip, 23 of them to
    build ``criteria_analysis`` and 4 to build ``trading_decision`` — which is the
    whole of the 27.3 s / 4.1 s history latency those two threads showed. It is
    also why the cost looked flat in ``limit``: it is paid before a single
    checkpoint row is read.

    Caching is safe because the tools are session-free: ``MultiServerMCPClient``
    documents that "a new session will be created for each tool call", so a cached
    tool holds a connection *spec*, not a live connection.
    """
    key = json.dumps(connections, sort_keys=True, default=str)

    def fresh() -> list | None:
        hit = _tool_cache.get(key)
        if hit and time.monotonic() - hit[0] < _TOOL_CACHE_TTL_SECONDS:
            return hit[1]
        return None

    if (tools := fresh()) is not None:
        return tools

    # One lock per connection set: a graph build fires this many times over, and
    # without it every one of them would open its own session before the first
    # result landed. Waiters re-check the cache rather than re-fetching.
    lock = _tool_cache_locks.setdefault(key, asyncio.Lock())
    async with lock:
        if (tools := fresh()) is not None:
            return tools
        # A failure is deliberately NOT cached — the next caller retries.
        tools = await MultiServerMCPClient(connections).get_tools()
        _tool_cache[key] = (time.monotonic(), tools)
        return tools


async def get_tools(
    config: RunnableConfig,
    allowed_tools: list[str],
    custom_tools: list | None = None,
) -> list:
    """Load MCP tools filtered to *allowed_tools*, plus any custom tools.

    Skips the MCP connection entirely when *allowed_tools* is empty, which
    avoids an unnecessary network round-trip for agents that only use custom
    tools (e.g. the web-search agent).
    """
    if allowed_tools:
        mcp_config = McpConfiguration.from_runnable_config(config)
        all_tools = await _load_all_mcp_tools(mcp_config.get_mcp_connections())
        allowed = set(allowed_tools)
        mcp_tools = [t for t in all_tools if t.name in allowed]
    else:
        mcp_tools = []
    return mcp_tools + (custom_tools or [])
