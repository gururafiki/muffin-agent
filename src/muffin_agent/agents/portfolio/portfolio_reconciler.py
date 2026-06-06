"""Portfolio reconciler — hybrid deterministic + LLM order producer.

Final step before trade execution.  Takes:

* ``portfolio`` — current state (cash, positions, margin)
* ``position_limits`` — output of :func:`position_sizing_node`
* ``ticker_decisions`` — one per ticker from :func:`ticker_decision_node`
* ``current_prices`` — for share-count math

…computes a deterministic ``allowed_actions`` per ticker (max buy / sell /
short / cover shares within constraints), pre-fills ``hold`` for any
ticker with no valid non-hold action (saves LLM cost), and asks the LLM
to pick concrete share counts from the legal menu.
"""

from __future__ import annotations

import logging
from typing import Any, cast

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from ...model_config import ModelConfiguration
from ...portfolio.executor import PortfolioOrder
from ...portfolio.state import Portfolio, mark_to_market
from ...prompts import render_template

logger = logging.getLogger(__name__)


class PortfolioReconciliationOutput(BaseModel):
    """Final orders + cross-ticker observations."""

    orders: list[PortfolioOrder] = Field(default_factory=list)
    """One order per ticker (including ``hold``).  Validated by the
    executor downstream — illegal orders are silently dropped at
    execution time."""

    portfolio_notes: str
    """1–3 sentences on cross-ticker context: concentration, sector
    overlap, total NAV deployed, etc."""


class PortfolioReconcilerInputState(TypedDict, total=False):
    """State keys read by ``portfolio_reconciler_node``."""

    portfolio: dict[str, Any]
    position_limits: dict[str, dict[str, Any]]
    ticker_decisions: dict[str, dict[str, Any]]
    current_prices: dict[str, float]


class PortfolioReconcilerOutputState(TypedDict, total=False):
    """State keys written by ``portfolio_reconciler_node``."""

    orders: list[dict[str, Any]]
    """List of ``PortfolioOrder.model_dump()`` dicts."""

    portfolio_notes: str


def _allowed_actions(
    portfolio: Portfolio,
    ticker: str,
    price: float,
    limit_dollars: float,
    remaining_dollars: float,
) -> dict[str, int]:
    """Compute the maximum share count for each legal action on *ticker*.

    Returns ``{"buy": N, "sell": N, "short": N, "cover": N, "hold": 0}``
    — ``hold`` is always available with quantity 0.  Other actions are
    capped at their constraint:

    * ``buy``: ``min(cash // price, remaining_dollars // price)``
    * ``sell``: existing long share count
    * ``short``: ``min(margin_avail // (price × margin_req), remaining // price)``
    * ``cover``: existing short share count
    """
    pos = portfolio.positions.get(ticker)
    long_qty = pos.long if pos else 0
    short_qty = pos.short if pos else 0

    buy_by_cash = int(portfolio.cash // price) if price > 0 else 0
    buy_by_limit = int(remaining_dollars // price) if price > 0 else 0
    max_buy = max(0, min(buy_by_cash, buy_by_limit))

    margin_req = portfolio.margin_requirement
    available_cash = max(0.0, portfolio.cash - portfolio.margin_used)
    if margin_req > 0 and price > 0:
        short_by_margin = int(available_cash / (price * margin_req))
    else:
        short_by_margin = 0
    short_by_limit = int(remaining_dollars // price) if price > 0 else 0
    max_short = max(0, min(short_by_margin, short_by_limit))

    return {
        "buy": max_buy,
        "sell": long_qty,
        "short": max_short,
        "cover": short_qty,
        "hold": 0,
    }


async def portfolio_reconciler_node(
    state: PortfolioReconcilerInputState, config: RunnableConfig
) -> PortfolioReconcilerOutputState:
    """Reconcile per-ticker decisions into concrete portfolio orders.

    Workflow:
      1. **Deterministic pre-pass** — compute ``allowed_actions`` per
         ticker; pre-fill ``hold`` for tickers where every non-hold
         action is impossible (saves the LLM compute).
      2. **LLM call** (skipped entirely when all tickers are pre-filled)
         with a compact representation of decisions + allowed actions
         + current prices.  Returns a ``PortfolioReconciliationOutput``
         carrying one ``PortfolioOrder`` per ticker.
    """
    portfolio_dump = state.get("portfolio") or {}
    limits = state.get("position_limits") or {}
    decisions = state.get("ticker_decisions") or {}
    prices = state.get("current_prices") or {}

    if not decisions:
        return {"orders": [], "portfolio_notes": "No ticker decisions to reconcile"}

    try:
        portfolio = Portfolio.model_validate(portfolio_dump)
    except Exception:
        logger.exception("portfolio_reconciler_node: invalid portfolio payload")
        # Default to all-holds when state is corrupt
        orders = [
            PortfolioOrder(
                ticker=t,
                action="hold",
                quantity=0,
                confidence=0.0,
                reasoning="Invalid portfolio state — defaulting to hold",
            )
            for t in decisions
        ]
        return {
            "orders": [o.model_dump() for o in orders],
            "portfolio_notes": "Portfolio state could not be parsed.",
        }

    portfolio_value = mark_to_market(portfolio, prices)
    nav = portfolio_value.nav

    # Pre-pass: build allowed_actions + pre-fill obvious holds
    allowed_by_ticker: dict[str, dict[str, int]] = {}
    pre_filled_orders: list[PortfolioOrder] = []
    llm_inputs: dict[str, dict[str, Any]] = {}
    for ticker, decision in decisions.items():
        price = prices.get(ticker)
        if price is None or price <= 0:
            pre_filled_orders.append(
                PortfolioOrder(
                    ticker=ticker,
                    action="hold",
                    quantity=0,
                    confidence=0.0,
                    reasoning="No current price available",
                )
            )
            continue
        limit = limits.get(ticker, {})
        limit_dollars = float(limit.get("limit_dollars") or 0.0)
        remaining_dollars = float(limit.get("remaining_dollars") or limit_dollars)
        allowed = _allowed_actions(
            portfolio, ticker, price, limit_dollars, remaining_dollars
        )
        allowed_by_ticker[ticker] = allowed

        # If every non-hold action has 0 capacity, pre-fill hold without LLM
        if all(allowed[k] == 0 for k in ("buy", "sell", "short", "cover")):
            pre_filled_orders.append(
                PortfolioOrder(
                    ticker=ticker,
                    action="hold",
                    quantity=0,
                    confidence=float(decision.get("confidence") or 0.5),
                    reasoning="No legal action available given current constraints",
                )
            )
            continue

        llm_inputs[ticker] = {
            "decision": decision,
            "allowed": allowed,
            "price": price,
            "limit_dollars": limit_dollars,
            "remaining_dollars": remaining_dollars,
        }

    # If everything is pre-filled, skip the LLM entirely
    if not llm_inputs:
        return {
            "orders": [o.model_dump() for o in pre_filled_orders],
            "portfolio_notes": (
                f"All tickers pre-filled to hold "
                f"(NAV ${nav:,.0f}, cash ${portfolio.cash:,.0f})."
            ),
        }

    llm = ModelConfiguration.get_chat_model_for_role(
        config, "reasoner", schema=PortfolioReconciliationOutput
    )
    prompt = render_template(
        "portfolio/portfolio_reconciler.jinja",
        nav=nav,
        cash=portfolio.cash,
        margin_used=portfolio.margin_used,
        margin_requirement=portfolio.margin_requirement,
        llm_inputs=llm_inputs,
        pre_filled_orders=[o.model_dump() for o in pre_filled_orders],
    )
    result = cast(
        PortfolioReconciliationOutput,
        await llm.ainvoke(
            [
                SystemMessage(prompt),
                HumanMessage("Produce the reconciled order list now."),
            ]
        ),
    )

    # Merge pre-filled holds with LLM-produced orders; ensure every
    # ticker in decisions has exactly one order.
    by_ticker: dict[str, PortfolioOrder] = {o.ticker: o for o in pre_filled_orders}
    for order in result.orders:
        by_ticker[order.ticker] = order
    # Fill any missing tickers (LLM may have dropped some) with hold
    for ticker in decisions:
        if ticker not in by_ticker:
            by_ticker[ticker] = PortfolioOrder(
                ticker=ticker,
                action="hold",
                quantity=0,
                confidence=0.0,
                reasoning="LLM did not produce an order — defaulting to hold",
            )

    final_orders = [by_ticker[t] for t in decisions]
    return {
        "orders": [o.model_dump() for o in final_orders],
        "portfolio_notes": result.portfolio_notes,
    }
