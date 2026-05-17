"""Fetch realised price outcomes for past trading decisions.

Exposes a single async callable ``fetch_outcomes_openbb(...)`` that:

1. Calls the OpenBB MCP ``equity_price_historical`` tool for both the
   decision ticker and the benchmark (default ``SPY``).
2. Computes raw return over the requested holding window and alpha vs the
   benchmark.
3. Returns ``None`` if either ticker has insufficient data for the window
   (decision too recent, ticker delisted, MCP unavailable). The caller
   treats ``None`` as "leave entry pending — try again next run".

Callers can substitute any callable with the same shape to plug in their
own data source (yfinance, internal market-data service, test fixtures).
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Protocol

from langchain_core.runnables import RunnableConfig

from ....mcp_config import McpConfiguration
from ..schemas import Outcome

logger = logging.getLogger(__name__)


class OutcomesFetcher(Protocol):
    """Async callable returning realised price outcomes (or ``None``)."""

    async def __call__(
        self,
        *,
        config: RunnableConfig,
        ticker: str,
        decision_date: str,
        holding_days: int,
        benchmark: str,
        decision_action: str | None,
    ) -> Outcome | None:
        """Return realised outcome for the (ticker, decision_date) pair or None."""
        ...


def _add_calendar_buffer(date_str: str, days: int) -> str:
    """Return ``date_str`` shifted forward by a calendar-day buffer.

    Trading-day fetches need a calendar-day window large enough to span
    the requested *holding_days* trading days plus weekends/holidays. A
    1.8× multiplier handles typical windows generously without ballooning
    request size.
    """
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    shifted = dt + timedelta(days=int(days * 1.8) + 3)
    return shifted.strftime("%Y-%m-%d")


def _extract_closes(raw: object) -> list[tuple[str, float]]:
    """Pull (date, close_price) tuples from a heterogeneous OpenBB payload.

    OpenBB MCP tools return different shapes across versions and providers;
    this helper accepts dicts, lists of dicts, and JSON-encoded variants
    and reduces them to a sorted list of ``(YYYY-MM-DD, close)`` rows.
    Returns an empty list on anything unrecognisable.
    """
    rows: list[dict] = []
    payload: object = raw
    if isinstance(payload, dict):
        for key in ("results", "data", "items"):
            if key in payload and isinstance(payload[key], list):
                payload = payload[key]
                break
    if isinstance(payload, list):
        rows = [r for r in payload if isinstance(r, dict)]

    out: list[tuple[str, float]] = []
    for row in rows:
        date_val = row.get("date") or row.get("Date") or row.get("timestamp")
        close_val = (
            row.get("close")
            or row.get("Close")
            or row.get("adj_close")
            or row.get("adjClose")
        )
        if not isinstance(date_val, str) or not isinstance(close_val, (int, float)):
            continue
        # Normalise to YYYY-MM-DD prefix.
        out.append((date_val[:10], float(close_val)))
    return sorted(out, key=lambda x: x[0])


def _window_return(
    closes: list[tuple[str, float]], decision_date: str, holding_days: int
) -> tuple[float, int] | None:
    """Compute (return_pct, actual_holding_days) over the holding window."""
    # First trading day >= decision_date.
    starts = [(d, p) for d, p in closes if d >= decision_date]
    if len(starts) < 2:
        return None
    start_date, start_price = starts[0]
    # Last available trading day within the holding window.
    end_idx = min(holding_days, len(starts) - 1)
    end_date, end_price = starts[end_idx]
    if start_price <= 0:
        return None
    return_pct = (end_price - start_price) / start_price * 100.0
    # Actual trading-day distance (best-effort — index difference).
    actual_days = end_idx
    return return_pct, actual_days


async def fetch_outcomes_openbb(
    *,
    config: RunnableConfig,
    ticker: str,
    decision_date: str,
    holding_days: int,
    benchmark: str,
    decision_action: str | None,
) -> Outcome | None:
    """Fetch realised return + alpha via the OpenBB MCP server.

    Returns ``None`` (so the caller leaves the entry pending) when:

    * Either ticker has fewer than 2 trading days at/after ``decision_date``.
    * The OpenBB MCP is unreachable.
    * The returned payload cannot be parsed into ``(date, close)`` rows.
    """
    end_date = _add_calendar_buffer(decision_date, holding_days)

    try:
        # Import locally so test-time monkeypatches of `fetch_outcomes_openbb`
        # avoid pulling in `langchain_mcp_adapters` (which would force an
        # MCP server connection at import time).
        from langchain_mcp_adapters.client import MultiServerMCPClient

        mcp_config = McpConfiguration.from_runnable_config(config)
        connections = mcp_config.get_mcp_connections()
        client = MultiServerMCPClient(connections=connections)
        tools = await client.get_tools()
        price_tool = next(
            (t for t in tools if t.name == "equity_price_historical"), None
        )
        if price_tool is None:
            logger.debug("fetch_outcomes_openbb: equity_price_historical not available")
            return None

        ticker_raw = await price_tool.ainvoke(
            {"symbol": ticker, "start_date": decision_date, "end_date": end_date}
        )
        bench_raw = await price_tool.ainvoke(
            {"symbol": benchmark, "start_date": decision_date, "end_date": end_date}
        )
    except Exception:
        logger.debug(
            "fetch_outcomes_openbb: MCP call failed for %s", ticker, exc_info=True
        )
        return None

    ticker_closes = _extract_closes(ticker_raw)
    bench_closes = _extract_closes(bench_raw)
    if not ticker_closes or not bench_closes:
        return None

    ticker_window = _window_return(ticker_closes, decision_date, holding_days)
    bench_window = _window_return(bench_closes, decision_date, holding_days)
    if ticker_window is None or bench_window is None:
        return None

    raw_return, actual_days = ticker_window
    bench_return, _ = bench_window

    return Outcome(
        raw_return_pct=round(raw_return, 4),
        alpha_return_pct=round(raw_return - bench_return, 4),
        holding_days=max(actual_days, 1),
        benchmark=benchmark,
        decision_action=decision_action,
    )
