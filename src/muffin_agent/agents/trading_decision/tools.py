"""In-process tools for trading_decision analysts and reflection bookends.

Two entry points co-located here because both pull daily OHLCV via the
existing OpenBB ``equity_price_historical`` MCP tool and reuse the same
``_extract_results_rows`` / ``_normalise_for_stockstats`` parsing helpers:

* :func:`get_indicators` (``@tool``) — called by the Market analyst.
  OpenBB MCP confirmed not to ship technical-indicator computation, so
  this fills the gap by computing the requested indicator locally via
  ``stockstats``.
* :func:`fetch_decision_outcome` (plain async) — called by
  ``reflection/resolver.py`` to realise the return + alpha vs benchmark
  for a past trading decision. Implements the default
  :class:`OutcomesFetcher` Protocol. Not ``@tool``-decorated because it's
  a deterministic helper for graph nodes, not LLM-callable — and adding
  ``@tool`` would wrap it in ``BaseTool`` and break the Protocol match.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from typing import Annotated, Any, Protocol

import pandas as pd  # type: ignore[import-untyped]
import stockstats  # type: ignore[import-untyped]
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.tools import InjectedToolArg, ToolException, tool

from ..data_collection.utils import get_tools
from .schemas import Outcome

logger = logging.getLogger(__name__)

# Indicators supported by the underlying ``stockstats`` library that are
# useful for short-/medium-term trend, momentum, and volatility reads.
# Keeping the surface tight prevents the model from asking for exotic
# indicators that ``stockstats`` may handle inconsistently.
_SUPPORTED_INDICATORS: frozenset[str] = frozenset(
    {
        "close_50_sma",
        "close_200_sma",
        "close_10_ema",
        "macd",
        "macds",
        "macdh",
        "rsi",
        "boll",
        "boll_ub",
        "boll_lb",
        "atr",
        "vwma",
    }
)

# History window pulled from OpenBB. Set generously so SMA-200 and other
# long-window indicators always have enough history to converge.
_HISTORY_DAYS_MIN: int = 300


@tool(parse_docstring=True)
async def get_indicators(
    ticker: str,
    indicator: str,
    curr_date: str,
    look_back_days: int = 30,
    config: Annotated[RunnableConfig | None, InjectedToolArg] = None,
) -> str:
    """Compute a single technical indicator over a rolling lookback window.

    Pulls daily OHLCV history from the OpenBB ``equity_price_historical``
    MCP tool and computes the indicator locally via ``stockstats``.

    Args:
        ticker: Ticker symbol of the company (e.g. ``AAPL``).
        indicator: Indicator name. Supported: ``close_50_sma``,
            ``close_200_sma``, ``close_10_ema``, ``macd``, ``macds``,
            ``macdh``, ``rsi``, ``boll``, ``boll_ub``, ``boll_lb``,
            ``atr``, ``vwma``.
        curr_date: End date of the lookback window (YYYY-MM-DD).
        look_back_days: Window size in trading days. Default 30.
        config: Injected by LangChain at runtime; carries OpenBB MCP
            credentials.

    Returns:
        A markdown table with ``date`` and ``<indicator>`` columns for
        the most recent ``look_back_days`` rows up to ``curr_date``.
    """
    indicator = indicator.strip().lower()
    if indicator not in _SUPPORTED_INDICATORS:
        return (
            f"Unsupported indicator: {indicator}. "
            f"Supported indicators: {sorted(_SUPPORTED_INDICATORS)}"
        )

    df = await _fetch_ohlcv_via_openbb(ticker, curr_date, look_back_days, config)
    if df.empty:
        return f"No OHLCV data returned for {ticker} up to {curr_date}."

    sdf = stockstats.wrap(df)
    sdf[indicator]  # trigger lazy stockstats computation
    series = sdf[indicator].dropna()
    if series.empty:
        return (
            f"Indicator {indicator} produced no values — likely insufficient "
            f"history for {ticker} as of {curr_date}."
        )
    tail = series.tail(look_back_days)
    return tail.to_frame(name=indicator).to_markdown()


async def _fetch_ohlcv_via_openbb(
    ticker: str,
    end_date: str,
    look_back_days: int,
    config: RunnableConfig | None,
) -> pd.DataFrame:
    """Pull OHLCV from OpenBB MCP and return a DataFrame normalised for stockstats."""
    tools = await get_tools(config or {}, ["equity_price_historical"])
    if not tools:
        raise ToolException(
            "equity_price_historical not available — is the OpenBB MCP server running?"
        )
    fetcher = tools[0]

    end_dt = datetime.strptime(end_date, "%Y-%m-%d").date()
    # Always request at least _HISTORY_DAYS_MIN of history so long-window
    # indicators (SMA-200) converge regardless of look_back_days.
    history_days = max(look_back_days * 3, _HISTORY_DAYS_MIN)
    start_dt = end_dt - timedelta(days=history_days)

    raw = await fetcher.ainvoke(
        {
            "provider": "yfinance",
            "symbol": ticker,
            "start_date": start_dt.isoformat(),
            "end_date": end_dt.isoformat(),
            "interval": "1d",
        }
    )
    rows = _extract_results_rows(raw)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    return _normalise_for_stockstats(df)


def _extract_results_rows(raw: Any) -> list[dict[str, Any]]:
    """Pull the ``results`` list out of an OpenBB tool response (str or dict)."""
    payload: Any = raw
    if isinstance(raw, str):
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ToolException(
                f"OpenBB equity_price_historical returned non-JSON response: "
                f"{str(raw)[:200]}"
            ) from exc
    if isinstance(payload, dict):
        results = payload.get("results")
        if isinstance(results, list):
            return [r for r in results if isinstance(r, dict)]
    if isinstance(payload, list):
        return [r for r in payload if isinstance(r, dict)]
    return []


def _normalise_for_stockstats(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce OpenBB OHLCV columns into the case + dtypes stockstats expects."""
    # OpenBB returns lower-case column names; stockstats also accepts
    # lower-case but it expects a ``date`` column for time-series ops.
    columns_lower = {c: c.lower() for c in df.columns}
    df = df.rename(columns=columns_lower)
    if "date" not in df.columns:
        raise ToolException(
            "OpenBB equity_price_historical response missing 'date' column."
        )
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    numeric_cols = [
        c for c in ("open", "high", "low", "close", "volume") if c in df.columns
    ]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=["close"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


# ── Outcome fetcher ─────────────────────────────────────────────────────────


class OutcomesFetcher(Protocol):
    """Async callable returning realised price outcomes for a past decision.

    Default implementation is :func:`fetch_decision_outcome`. Tests and
    alternative price sources (yfinance direct, internal market-data
    service, cached fixtures) plug in by passing a different callable to
    ``reflector_resolve_node(..., outcomes_fetcher=...)``.
    """

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


def _add_calendar_buffer(date_str: str, trading_days: int) -> str:
    """Pad a trading-day window to a calendar-day window that spans weekends.

    ~1.8× factor + 3-day baseline covers normal holding windows generously
    without ballooning the request size.
    """
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    shifted = dt + timedelta(days=int(trading_days * 1.8) + 3)
    return shifted.strftime("%Y-%m-%d")


def _extract_closes(rows: list[dict[str, Any]]) -> list[tuple[str, float]]:
    """Pick ``(date, close)`` tuples from already-parsed OpenBB row dicts.

    Tolerates column-name variation OpenBB returns across providers
    (``close`` / ``Close`` / ``adj_close`` / ``adjClose``). Sorted ascending;
    empty list when no usable rows.
    """
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
        out.append((date_val[:10], float(close_val)))
    return sorted(out, key=lambda x: x[0])


def _window_return(
    closes: list[tuple[str, float]], decision_date: str, holding_days: int
) -> tuple[float, int] | None:
    """Compute ``(return_pct, actual_trading_days)`` over the holding window."""
    starts = [(d, p) for d, p in closes if d >= decision_date]
    if len(starts) < 2:
        return None
    _, start_price = starts[0]
    end_idx = min(holding_days, len(starts) - 1)
    _, end_price = starts[end_idx]
    if start_price <= 0:
        return None
    return (end_price - start_price) / start_price * 100.0, end_idx


async def _fetch_closes(
    price_fetcher: Runnable[dict[str, Any], Any],
    symbol: str,
    start_date: str,
    end_date: str,
) -> list[tuple[str, float]]:
    """Pull OHLCV for *symbol* from OpenBB and return sorted ``(date, close)``.

    Returns an empty list on parse failure (caller treats empty as "no data").
    """
    raw = await price_fetcher.ainvoke(
        {"symbol": symbol, "start_date": start_date, "end_date": end_date}
    )
    return _extract_closes(_extract_results_rows(raw))


async def fetch_decision_outcome(
    *,
    config: RunnableConfig,
    ticker: str,
    decision_date: str,
    holding_days: int = 5,
    benchmark: str = "SPY",
    decision_action: str | None = None,
) -> Outcome | None:
    """Compute realised return + alpha vs benchmark for a past trading decision.

    Default :class:`OutcomesFetcher` implementation. Pulls daily OHLCV for
    ``ticker`` and ``benchmark`` from OpenBB's ``equity_price_historical``
    and returns the (return_pct, alpha_pct) over the first ``holding_days``
    trading days at/after ``decision_date``. Returns ``None`` (caller
    leaves the decision pending) when:

    * Either symbol has fewer than 2 trading days at/after ``decision_date``.
    * OpenBB MCP is unreachable.
    * The payload cannot be parsed.

    Keyword-only signature matches the :class:`OutcomesFetcher` Protocol
    so callers can swap implementations transparently. Not decorated with
    ``@tool`` because this is a deterministic helper called by graph nodes
    rather than an LLM-callable tool — adding ``@tool`` would wrap the
    function in ``BaseTool`` and break the Protocol match.

    Args:
        config: LangChain runnable config carrying OpenBB MCP credentials.
        ticker: Symbol; preserve exchange suffix (``AAPL`` / ``TSM.TO``).
        decision_date: Anchor date ``YYYY-MM-DD``; window starts on the
            first trading day at/after this date.
        holding_days: Trading-day window length. Default 5.
        benchmark: Benchmark ticker for alpha calculation. Default ``SPY``.
        decision_action: Optional rating from the original decision; passed
            through into the returned ``Outcome`` for downstream prompts.

    Returns:
        :class:`Outcome` with raw return %, alpha %, and actual holding days;
        ``None`` if the window can't be computed yet.
    """
    end_date = _add_calendar_buffer(decision_date, holding_days)
    tools = await get_tools(config or {}, ["equity_price_historical"])
    if not tools:
        logger.debug("fetch_decision_outcome: equity_price_historical not available")
        return None
    price_fetcher = tools[0]

    try:
        ticker_closes = await _fetch_closes(
            price_fetcher, ticker, decision_date, end_date
        )
        bench_closes = await _fetch_closes(
            price_fetcher, benchmark, decision_date, end_date
        )
    except Exception:
        logger.debug(
            "fetch_decision_outcome MCP call failed for %s", ticker, exc_info=True
        )
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
