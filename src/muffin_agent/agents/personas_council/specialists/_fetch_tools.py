"""Deterministic MCP fetches expressed as REAL tool calls.

The deterministic specialists (``technical_analysis``, ``sentiment_analysis``) used
to call :func:`cached_invoke` straight from a graph node. That works, but it bypasses
the message machinery entirely — no ``AIMessage``/``ToolMessage`` pair is ever
produced, so the fetch is invisible to every consumer that reads a namespace's
``values.messages`` (the UI execution tree, LangSmith trajectory views, evals).

This module keeps execution **fully deterministic** (no LLM decides anything — the
args are computed in Python) while producing genuine tool-call records, by pairing:

* the ``fetch_*`` tools below, which resolve their MCP tool at runtime and go through
  :func:`cached_invoke` so cache keys still collide perfectly with the persona /
  ``ToolResultCacheMiddleware`` path (namespace ``("cache", <mcp tool name>)`` +
  ``get_args_hash(args)``) — note the explicit ``tool_name=`` override, without which
  the wrapper's own name would fork the cache; and
* :func:`deterministic_tool_call` / :func:`plan_message`, which build the synthetic
  ``AIMessage`` a bare :class:`~langgraph.prebuilt.ToolNode` consumes.

``ToolNode`` documents this exact usage — its input formats include *"Direct Tool
Calls"* and its docstring scopes it to "custom routing logic … non-standard agent
architectures". The resulting subgraph is ``plan_fetch -> ToolNode -> compute``.

A failed fetch now raises, so ``ToolNode`` records an ``status="error"`` ToolMessage
instead of the old silent empty-list return — the failure becomes visible rather than
being indistinguishable from "no data existed".
"""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import Sequence
from typing import Any

from langchain_core.messages import AIMessage, AnyMessage, ToolMessage
from langchain_core.messages.tool import ToolCall
from langchain_core.tools import ToolException, tool
from langgraph.config import get_config, get_store

from ....middlewares.tool_result_cache import cached_invoke
from ...data_collection.utils import get_tools

logger = logging.getLogger(__name__)


# ── Plumbing ──────────────────────────────────────────────────────────────────


async def _cached_mcp_fetch(
    mcp_tool_name: str, args: dict[str, Any]
) -> str | list[Any]:
    """Resolve *mcp_tool_name* from MCP and invoke it through the shared cache.

    Args:
        mcp_tool_name: The OpenBB/Firecrawl MCP tool to call.
        args: Tool args, passed verbatim (and hashed for the cache key).

    Returns:
        The raw result content (``str`` or ``list`` — the ``ToolMessage.content``
        shape).

    Raises:
        ToolException: When the MCP tool is unavailable or the call fails. Surfaces as
            an ``status="error"`` ToolMessage rather than crashing the graph.
    """
    tools = await get_tools(get_config(), [mcp_tool_name])
    if not tools:
        msg = f"MCP tool {mcp_tool_name!r} is not available"
        raise ToolException(msg)
    try:
        store = get_store()
    except Exception:  # no store configured (e.g. bare unit test)
        store = None
    try:
        return await cached_invoke(tools[0], args, store, tool_name=mcp_tool_name)
    except Exception as exc:
        logger.exception("deterministic fetch %s failed", mcp_tool_name)
        msg = f"{mcp_tool_name} failed: {exc}"
        raise ToolException(msg) from exc


def deterministic_tool_call(name: str, args: dict[str, Any]) -> ToolCall:
    """Build one synthetic ``ToolCall`` for a bare ``ToolNode`` to execute."""
    return {
        "name": name,
        "args": args,
        "id": f"det_{uuid.uuid4().hex[:12]}",
        "type": "tool_call",
    }


def plan_message(calls: Sequence[ToolCall], *, note: str = "") -> AIMessage:
    """Wrap *calls* in the ``AIMessage`` a ``ToolNode`` reads its work from.

    Args:
        calls: The tool calls to execute. Several calls in ONE message are executed
            concurrently by ``ToolNode`` — that is how the sentiment specialist keeps
            its insider + news fetches parallel without two graph nodes.
        note: Optional human-readable content for the message.
    """
    return AIMessage(content=note, tool_calls=list(calls))


def tool_payload(messages: Sequence[AnyMessage] | None, tool_name: str) -> Any:
    """Return the most recent successful ``ToolMessage`` content for *tool_name*."""
    for message in reversed(list(messages or [])):
        if (
            isinstance(message, ToolMessage)
            and message.name == tool_name
            and message.status != "error"
        ):
            return message.content
    return None


def parse_result_rows(raw: Any) -> list[dict[str, Any]]:
    """Extract row dicts from an OpenBB-style response (``{"results": [...]}``)."""
    payload: Any = raw
    if isinstance(raw, str):
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return []
    if isinstance(payload, dict):
        results = payload.get("results")
        if isinstance(results, list):
            return [r for r in results if isinstance(r, dict)]
    if isinstance(payload, list):
        return [r for r in payload if isinstance(r, dict)]
    return []


# ── The deterministic fetch tools ─────────────────────────────────────────────


@tool(parse_docstring=True)
async def fetch_equity_ohlcv(
    symbol: str,
    start_date: str,
    end_date: str,
    interval: str = "1d",
    provider: str = "yfinance",
) -> str | list[Any]:
    """Fetch historical OHLCV price bars for one equity.

    Args:
        symbol: Ticker symbol, exchange suffix preserved (e.g. ``AAPL``, ``SHOP.TO``).
        start_date: Inclusive ISO start date (``YYYY-MM-DD``).
        end_date: Inclusive ISO end date (``YYYY-MM-DD``).
        interval: Bar interval; ``1d`` for daily bars.
        provider: OpenBB data provider.

    Returns:
        The raw provider payload containing a ``results`` array of OHLCV bars.
    """
    return await _cached_mcp_fetch(
        "equity_price_historical",
        {
            "provider": provider,
            "symbol": symbol,
            "start_date": start_date,
            "end_date": end_date,
            "interval": interval,
        },
    )


@tool(parse_docstring=True)
async def fetch_insider_trades(symbol: str, limit: int = 100) -> str | list[Any]:
    """Fetch recent insider (Form 4) transactions for one equity.

    Args:
        symbol: Ticker symbol, exchange suffix preserved.
        limit: Maximum number of transactions to return.

    Returns:
        The raw provider payload containing a ``results`` array of insider trades,
        each with a signed ``transaction_shares`` field.
    """
    return await _cached_mcp_fetch(
        "equity_ownership_insider_trading", {"symbol": symbol, "limit": limit}
    )


@tool(parse_docstring=True)
async def fetch_company_news(
    symbol: str,
    start_date: str,
    end_date: str,
    limit: int = 50,
    provider: str = "benzinga",
) -> str | list[Any]:
    """Fetch company news articles carrying a per-article sentiment label.

    ``benzinga`` is the only OpenBB news provider that consistently exposes a
    per-article ``sentiment`` field, which the deterministic aggregation reads.

    Args:
        symbol: Ticker symbol, exchange suffix preserved.
        start_date: Inclusive ISO start date (``YYYY-MM-DD``).
        end_date: Inclusive ISO end date (``YYYY-MM-DD``).
        limit: Maximum number of articles to return.
        provider: OpenBB news provider.

    Returns:
        The raw provider payload containing a ``results`` array of articles.
    """
    return await _cached_mcp_fetch(
        "news_company",
        {
            "provider": provider,
            "symbol": symbol,
            "start_date": start_date,
            "end_date": end_date,
            "limit": limit,
        },
    )
