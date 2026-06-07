"""Multi-ticker paper-trading portfolio decision graph.

Composes the council from :mod:`agents.personas` with the new
portfolio-level nodes (position sizing, ticker decision, reconciler,
executor) to produce concrete portfolio orders + an updated portfolio
snapshot.

Topology::

    START
      │
      ▼
    [Send fan-out × N tickers]
      │
      ▼
    ┌──────────────────────────────────────────────────────────┐
    │  council_graph(per ticker) → ticker_decision_node        │
    │  → writes {ticker: ticker_decision} via per-key reducer  │
    └──────────────────────────────────────────────────────────┘
      │
      │  (fan-in: ticker_decisions accumulated via merge reducer)
      ▼
    risk_management
      │
      ▼
    portfolio_reconciler
      │
      ▼
    execute_orders        ← applies orders via portfolio.executor
      │
      ▼
    END

**Independent of trading_decision** — uses the persona council, not
Bull/Bear/Judge/Trader.  Each persona owns its own data fetch
inside its subgraph, so the orchestrator pulls ``prices_history``
separately via ``cached_invoke(equity_price_historical, …)`` — this
shares cache hits with the technicals specialist and any persona that
also fetched OHLCV during its data_collection step.

The portfolio-level steps (sizing / reconciliation / execution) run
once across all tickers after the per-ticker subgraphs complete.
"""

from __future__ import annotations

import json
import logging
import operator
from datetime import UTC, datetime, timedelta
from typing import Annotated, Any

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.config import get_store
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.base import BaseStore
from langgraph.types import Send
from typing_extensions import TypedDict

from ....middlewares.tool_result_cache import cached_invoke
from ...data_collection.utils import get_tools
from ..council_graph import build_council_graph
from .executor import PortfolioOrder, apply_orders
from .portfolio_reconciler import portfolio_reconciler_node
from .risk_manager import risk_management_node
from .state import Portfolio, mark_to_market
from .ticker_decision import ticker_decision_node

logger = logging.getLogger(__name__)

_PRICE_LOOKBACK_DAYS = 365


def _parse_ohlcv_response(raw: Any) -> list[dict[str, Any]]:
    """Extract OHLCV bar dicts from an OpenBB ``equity_price_historical`` response."""
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


async def _fetch_prices_history(
    ticker: str, config: RunnableConfig
) -> list[dict[str, Any]]:
    """Pull 1y daily OHLCV via ``cached_invoke`` for position sizing.

    Uses the same args shape as ``specialists/technical_analysis.py``'s
    ``fetch_ohlcv_node`` so cache hits dedupe across the specialist + the
    portfolio-sizing fetch + any persona that fetched OHLCV during its
    own ``collect_data`` step.
    """
    if not ticker:
        return []
    try:
        store = get_store()
        tools = await get_tools(config, ["equity_price_historical"])
        if not tools:
            return []
        end_dt = datetime.now(UTC).date()
        start_dt = end_dt - timedelta(days=_PRICE_LOOKBACK_DAYS)
        raw = await cached_invoke(
            tools[0],
            {
                "provider": "yfinance",
                "symbol": ticker,
                "start_date": start_dt.isoformat(),
                "end_date": end_dt.isoformat(),
                "interval": "1d",
            },
            store,
        )
    except Exception:
        logger.exception("portfolio_decision fetch_prices failed for %s", ticker)
        return []
    return _parse_ohlcv_response(raw)


def _merge_dicts(a: dict[str, Any] | None, b: dict[str, Any] | None) -> dict[str, Any]:
    """Shallow-merge reducer used to accumulate per-ticker dicts.

    Used for ``ticker_decisions``, ``prices_history``, and
    ``per_ticker_signals`` — each fan-out worker writes ``{ticker:
    value}`` and the reducer merges across workers.
    """
    if a is None:
        return b or {}
    if b is None:
        return a
    out = dict(a)
    out.update(b)
    return out


class PortfolioDecisionState(TypedDict, total=False):
    """Top-level state for the multi-ticker portfolio decision graph."""

    # Inputs
    tickers: list[str]
    query: str
    portfolio: dict[str, Any]
    """``Portfolio.model_dump()`` snapshot.  Caller supplies the starting
    state; the graph returns a new ``portfolio`` snapshot at the end."""

    current_prices: dict[str, float]
    """Caller-supplied mark-to-market prices.  These are also derived
    from the per-ticker data bundles and merged for backwards-fill."""

    # Per-ticker outputs (accumulated via reducer)
    ticker_decisions: Annotated[dict[str, dict[str, Any]], _merge_dicts]
    prices_history: Annotated[dict[str, list[dict[str, Any]]], _merge_dicts]

    # Portfolio-level outputs
    position_limits: dict[str, dict[str, Any]]
    orders: list[dict[str, Any]]
    portfolio_notes: str
    executed_trades: list[dict[str, Any]]
    portfolio_value: dict[str, Any]
    """``PortfolioValue.model_dump()`` after orders execute."""


async def _build_per_ticker_subgraph(
    *, config: RunnableConfig | None = None, store: BaseStore | None = None
) -> CompiledStateGraph:
    """Inner subgraph: council → ticker_decision per single ticker.

    Compiled once at graph-build time; reused across all tickers via
    ``Send`` fan-out.  Returns a ``CompiledStateGraph`` whose state
    schema matches the ``Send`` payload below.
    """

    class _PerTickerState(TypedDict, total=False):
        ticker: str
        query: str
        persona_signals: Annotated[list[dict[str, Any]], operator.add]
        council_synthesis: dict[str, Any]
        ticker_decision: dict[str, Any]
        prices_history_for_ticker: list[dict[str, Any]]

    council = await build_council_graph(config or {}, store=store)

    async def _run_council(
        state: _PerTickerState, config: RunnableConfig
    ) -> dict[str, Any]:
        """Invoke the council graph + fetch OHLCV for position sizing.

        Each persona owns its own data fetch, but we still need 1y daily
        OHLCV downstream so ``risk_management_node`` can compute per-ticker
        annualised volatility + the cross-ticker correlation matrix.
        ``cached_invoke`` shares hits with the technicals specialist + any
        persona that also fetched OHLCV during its data_collection step.
        """
        ticker = state.get("ticker", "")
        result = await council.ainvoke(
            {"ticker": ticker, "query": state.get("query")},
            config,
        )
        prices_history = await _fetch_prices_history(ticker, config)
        return {
            "persona_signals": result.get("persona_signals", []),
            "council_synthesis": result.get("council_synthesis", {}),
            "prices_history_for_ticker": prices_history,
        }

    g: StateGraph = StateGraph(_PerTickerState)
    g.add_node("council", _run_council)
    g.add_node("ticker_decision", ticker_decision_node)
    g.add_edge(START, "council")
    g.add_edge("council", "ticker_decision")
    g.add_edge("ticker_decision", END)
    return g.compile()


async def build_portfolio_decision_graph(
    *,
    config: RunnableConfig | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
    store: BaseStore | None = None,
) -> CompiledStateGraph:
    """Build the multi-ticker paper-trading portfolio decision graph."""
    per_ticker = await _build_per_ticker_subgraph(config=config, store=store)

    class _AnalyzeTickerInput(TypedDict, total=False):
        """Send fan-out payload — one ticker + its inherited query."""

        ticker: str
        query: str

    async def _analyze_ticker(
        state: _AnalyzeTickerInput, config: RunnableConfig
    ) -> dict[str, Any]:
        """Run the per-ticker subgraph and lift its outputs into parent state."""
        ticker = state.get("ticker", "")
        result = await per_ticker.ainvoke(dict(state), config)
        return {
            "ticker_decisions": {ticker: result.get("ticker_decision") or {}},
            "prices_history": {ticker: result.get("prices_history_for_ticker") or []},
        }

    def _fanout_tickers(state: PortfolioDecisionState) -> list[Send]:
        return [
            Send(
                "analyze_ticker",
                {"ticker": t, "query": state.get("query")},
            )
            for t in state.get("tickers", [])
        ]

    async def _execute_orders(
        state: PortfolioDecisionState, config: RunnableConfig
    ) -> dict[str, Any]:
        """Apply the reconciled orders to the portfolio + mark-to-market."""
        portfolio_dump = state.get("portfolio") or {}
        orders_raw = state.get("orders") or []
        prices = state.get("current_prices") or {}
        try:
            portfolio = Portfolio.model_validate(portfolio_dump)
        except Exception:
            logger.exception("_execute_orders: invalid portfolio payload")
            return {"executed_trades": [], "portfolio_value": {}}

        # Decode orders into PortfolioOrder objects
        orders = [PortfolioOrder.model_validate(o) for o in orders_raw]
        new_portfolio, trades = apply_orders(portfolio, orders, prices)
        value = mark_to_market(new_portfolio, prices)
        return {
            "portfolio": new_portfolio.model_dump(),
            "executed_trades": [t.model_dump() for t in trades],
            "portfolio_value": value.model_dump(),
        }

    graph: StateGraph = StateGraph(PortfolioDecisionState)
    graph.add_node("analyze_ticker", _analyze_ticker)
    graph.add_node("risk_management", risk_management_node)
    graph.add_node("portfolio_reconciler", portfolio_reconciler_node)
    graph.add_node("execute_orders", _execute_orders)

    graph.add_conditional_edges(START, _fanout_tickers, ["analyze_ticker"])
    graph.add_edge("analyze_ticker", "risk_management")
    graph.add_edge("risk_management", "portfolio_reconciler")
    graph.add_edge("portfolio_reconciler", "execute_orders")
    graph.add_edge("execute_orders", END)

    return graph.compile(checkpointer=checkpointer, store=store)
