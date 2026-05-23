"""In-process tools for trading_decision analysts.

OpenBB MCP confirmed not to ship technical-indicator computation
tools — only macro `economy_indicators` are present. ``get_indicators``
fills the gap by pulling daily OHLCV via the existing OpenBB
``equity_price_historical`` MCP tool and computing the requested
indicator locally with ``stockstats``.

The Market analyst is the only consumer today; co-located here because
the tool is specific to the analyst layer in ``trading_decision``.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import Annotated, Any

import pandas as pd  # type: ignore[import-untyped]
import stockstats  # type: ignore[import-untyped]
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import InjectedToolArg, ToolException, tool

from ..data_collection.utils import get_tools

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
