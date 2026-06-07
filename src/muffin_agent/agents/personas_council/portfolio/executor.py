"""Trade execution — apply orders to a portfolio and produce a trade log.

Pure function: takes a portfolio + list of orders + current prices,
returns ``(new_portfolio, executed_trades)``.  Defensively validates
each order against the current portfolio state — illegal orders (e.g.
sell with no long position, cover with no short) are silently dropped
with a ``skipped`` flag in the trade log.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from .state import (
    Portfolio,
    apply_long_buy,
    apply_long_sell,
    apply_short_cover,
    apply_short_open,
)

PortfolioAction = Literal["buy", "sell", "short", "cover", "hold"]


class PortfolioOrder(BaseModel):
    """One trading order — produced by the portfolio reconciler."""

    ticker: str
    action: PortfolioAction
    quantity: int = Field(ge=0)
    """Share count.  ``0`` is valid for ``hold``."""

    confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    reasoning: str = ""


class ExecutedTrade(BaseModel):
    """One trade actually applied to the portfolio.

    ``executed_quantity`` may be less than ``order.quantity`` when the
    portfolio could only partially fill (insufficient cash / margin /
    existing position).  ``skipped`` is ``True`` when the order was
    rejected entirely (e.g. ``sell`` with no long position).
    """

    ticker: str
    action: PortfolioAction
    requested_quantity: int
    executed_quantity: int
    price: float
    skipped: bool = False
    reason: str = ""


def apply_orders(
    portfolio: Portfolio,
    orders: list[PortfolioOrder],
    prices: dict[str, float],
) -> tuple[Portfolio, list[ExecutedTrade]]:
    """Apply *orders* to *portfolio* at *prices*; return new state + trade log.

    Orders are applied in the supplied order.  Each long/short helper is
    called with the *current* portfolio state, so the partial-fill logic
    is consistent with what the upstream backtester does.

    Holds always pass through with zero execution and ``skipped=False``.
    Orders missing a price are skipped.

    Args:
        portfolio: Starting state.
        orders: List of :class:`PortfolioOrder` to apply.
        prices: Mapping of ticker → current price.

    Returns:
        ``(new_portfolio, executed_trades)`` — *new_portfolio* is a new
        instance with every successful execution applied; *executed_trades*
        carries one entry per order (including holds and skips).
    """
    p = portfolio
    trades: list[ExecutedTrade] = []
    for order in orders:
        price = prices.get(order.ticker)
        if order.action == "hold":
            trades.append(
                ExecutedTrade(
                    ticker=order.ticker,
                    action="hold",
                    requested_quantity=order.quantity,
                    executed_quantity=0,
                    price=price or 0.0,
                )
            )
            continue
        if price is None or price <= 0:
            trades.append(
                ExecutedTrade(
                    ticker=order.ticker,
                    action=order.action,
                    requested_quantity=order.quantity,
                    executed_quantity=0,
                    price=0.0,
                    skipped=True,
                    reason="No price available",
                )
            )
            continue

        if order.action == "buy":
            new_p, executed = apply_long_buy(p, order.ticker, order.quantity, price)
        elif order.action == "sell":
            new_p, executed = apply_long_sell(p, order.ticker, order.quantity, price)
        elif order.action == "short":
            new_p, executed = apply_short_open(p, order.ticker, order.quantity, price)
        elif order.action == "cover":
            new_p, executed = apply_short_cover(p, order.ticker, order.quantity, price)
        else:
            # Unknown action — shouldn't happen given the Literal
            trades.append(
                ExecutedTrade(
                    ticker=order.ticker,
                    action=order.action,
                    requested_quantity=order.quantity,
                    executed_quantity=0,
                    price=price,
                    skipped=True,
                    reason=f"Unknown action {order.action!r}",
                )
            )
            continue

        if executed == 0:
            trades.append(
                ExecutedTrade(
                    ticker=order.ticker,
                    action=order.action,
                    requested_quantity=order.quantity,
                    executed_quantity=0,
                    price=price,
                    skipped=True,
                    reason="Insufficient cash / margin / position",
                )
            )
        else:
            trades.append(
                ExecutedTrade(
                    ticker=order.ticker,
                    action=order.action,
                    requested_quantity=order.quantity,
                    executed_quantity=executed,
                    price=price,
                )
            )
            p = new_p

    return p, trades
