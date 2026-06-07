"""Portfolio state model + pure-function mutation helpers.

Ports ai-hedge-fund's mutable ``Portfolio`` class to muffin's idiomatic
LangGraph state: immutable Pydantic dataclasses, pure functions that
return new ``Portfolio`` instances.

Long / short bookkeeping (cost basis, realised gains, margin used) is
preserved exactly; the only structural difference is the return shape:
each mutation helper returns ``(new_portfolio, executed_quantity)`` so
the caller can detect partial fills.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Position(BaseModel):
    """Per-ticker position state.

    ``long_cost_basis`` is the weighted-average cost basis across all long
    buys (similarly for ``short_cost_basis``).  ``short_margin_used`` is
    the dollar margin locked up by this position's short legs.
    """

    long: int = 0
    short: int = 0
    long_cost_basis: float = 0.0
    short_cost_basis: float = 0.0
    short_margin_used: float = 0.0


class RealizedGain(BaseModel):
    """Realised P&L per ticker, split between long and short closes."""

    long: float = 0.0
    short: float = 0.0


class Portfolio(BaseModel):
    """Top-level portfolio state.

    Mutation helpers below return new ``Portfolio`` instances rather
    than mutating in place — friendlier for LangGraph state passing and
    consistent with how every other muffin state schema works.
    """

    cash: float
    margin_requirement: float = 0.5
    """Fraction of short proceeds that must be held as margin (0.5 = 50%)."""

    margin_used: float = 0.0
    positions: dict[str, Position] = Field(default_factory=dict)
    realized_gains: dict[str, RealizedGain] = Field(default_factory=dict)


class PortfolioValue(BaseModel):
    """Snapshot of mark-to-market portfolio value at given prices."""

    cash: float
    long_value: float
    short_exposure: float
    realized_gains_total: float
    nav: float
    """Net asset value = cash + long_value − short_exposure +
    realised gains."""

    margin_used: float


# ── Pure mutation helpers ─────────────────────────────────────────────────────


def _ensure_ticker(portfolio: Portfolio, ticker: str) -> None:
    """Initialise position + realised-gain entries for *ticker* if missing.

    Mutates *portfolio* in place — only called inside the mutation helpers
    where ``portfolio`` is already a fresh copy.
    """
    if ticker not in portfolio.positions:
        portfolio.positions[ticker] = Position()
    if ticker not in portfolio.realized_gains:
        portfolio.realized_gains[ticker] = RealizedGain()


def apply_long_buy(
    portfolio: Portfolio, ticker: str, quantity: int, price: float
) -> tuple[Portfolio, int]:
    """Buy *quantity* shares of *ticker* at *price*.

    Partial fill semantics: if cash is insufficient for *quantity* shares,
    fill the maximum affordable.  Returns ``(new_portfolio, executed_qty)``.
    """
    if quantity <= 0 or price <= 0:
        return portfolio, 0
    p = portfolio.model_copy(deep=True)
    _ensure_ticker(p, ticker)
    pos = p.positions[ticker]

    desired_cost = quantity * price
    if desired_cost <= p.cash:
        executed = int(quantity)
    else:
        executed = int(p.cash // price)
    if executed <= 0:
        return portfolio, 0

    cost = executed * price
    total_shares = pos.long + executed
    if total_shares > 0:
        pos.long_cost_basis = (pos.long_cost_basis * pos.long + cost) / total_shares
    pos.long = total_shares
    p.cash -= cost
    return p, executed


def apply_long_sell(
    portfolio: Portfolio, ticker: str, quantity: int, price: float
) -> tuple[Portfolio, int]:
    """Sell *quantity* shares of *ticker* at *price*.

    Sells at most ``position.long`` shares.  Realises gain
    ``(price − avg_cost) × executed`` into ``realized_gains[ticker].long``.
    """
    if quantity <= 0:
        return portfolio, 0
    p = portfolio.model_copy(deep=True)
    _ensure_ticker(p, ticker)
    pos = p.positions[ticker]
    executed = min(int(quantity), pos.long)
    if executed <= 0:
        return portfolio, 0

    avg_cost = pos.long_cost_basis if pos.long > 0 else 0.0
    realized = (price - avg_cost) * executed
    p.realized_gains[ticker].long += realized
    pos.long -= executed
    p.cash += executed * price
    if pos.long == 0:
        pos.long_cost_basis = 0.0
    return p, executed


def apply_short_open(
    portfolio: Portfolio, ticker: str, quantity: int, price: float
) -> tuple[Portfolio, int]:
    """Open a short of *quantity* shares at *price*.

    Locks ``proceeds × margin_requirement`` as margin.  Partial fill if
    available cash + margin would be exceeded.
    """
    if quantity <= 0 or price <= 0:
        return portfolio, 0
    p = portfolio.model_copy(deep=True)
    _ensure_ticker(p, ticker)
    pos = p.positions[ticker]
    margin_ratio = p.margin_requirement

    desired_proceeds = quantity * price
    desired_margin = desired_proceeds * margin_ratio
    available_cash = max(0.0, p.cash - p.margin_used)
    if desired_margin <= available_cash:
        executed = int(quantity)
    elif margin_ratio > 0:
        executed = int(available_cash / (price * margin_ratio))
    else:
        executed = 0
    if executed <= 0:
        return portfolio, 0

    proceeds = executed * price
    margin_required = proceeds * margin_ratio
    total_shorts = pos.short + executed
    if total_shorts > 0:
        pos.short_cost_basis = (
            pos.short_cost_basis * pos.short + proceeds
        ) / total_shorts
    pos.short = total_shorts
    pos.short_margin_used += margin_required
    p.margin_used += margin_required
    p.cash += proceeds
    p.cash -= margin_required
    return p, executed


def apply_short_cover(
    portfolio: Portfolio, ticker: str, quantity: int, price: float
) -> tuple[Portfolio, int]:
    """Cover *quantity* short shares at *price*.

    Caps at ``position.short``.  Releases proportional margin and
    realises ``(short_basis − price) × executed`` into
    ``realized_gains[ticker].short``.
    """
    if quantity <= 0:
        return portfolio, 0
    p = portfolio.model_copy(deep=True)
    _ensure_ticker(p, ticker)
    pos = p.positions[ticker]
    executed = min(int(quantity), pos.short)
    if executed <= 0:
        return portfolio, 0

    avg_short = pos.short_cost_basis if pos.short > 0 else 0.0
    realized = (avg_short - price) * executed
    portion = executed / pos.short if pos.short > 0 else 1.0
    margin_release = portion * pos.short_margin_used

    pos.short -= executed
    pos.short_margin_used -= margin_release
    p.margin_used -= margin_release
    p.cash += margin_release
    p.cash -= executed * price
    p.realized_gains[ticker].short += realized
    if pos.short == 0:
        pos.short_cost_basis = 0.0
        pos.short_margin_used = 0.0
    return p, executed


def mark_to_market(portfolio: Portfolio, prices: dict[str, float]) -> PortfolioValue:
    """Compute portfolio value at current *prices*.

    ``nav`` = cash + sum(long × price) − sum(short × price) + realised
    gains (realised gains are already in cash; included separately so
    the breakdown shows where NAV came from).

    Args:
        portfolio: Current portfolio state.
        prices: Mapping of ticker → current price.  Tickers absent from
            *prices* contribute zero to the long/short totals (defensive).

    Returns:
        :class:`PortfolioValue` with cash, long_value, short_exposure,
        realised gains total, NAV, and margin_used.
    """
    long_value = 0.0
    short_exposure = 0.0
    for ticker, pos in portfolio.positions.items():
        px = prices.get(ticker)
        if px is None:
            continue
        long_value += pos.long * px
        short_exposure += pos.short * px

    realised_total = sum(g.long + g.short for g in portfolio.realized_gains.values())
    nav = portfolio.cash + long_value - short_exposure
    return PortfolioValue(
        cash=portfolio.cash,
        long_value=long_value,
        short_exposure=short_exposure,
        realized_gains_total=realised_total,
        nav=nav,
        margin_used=portfolio.margin_used,
    )


def new_portfolio(*, initial_cash: float, margin_requirement: float = 0.5) -> Portfolio:
    """Construct a fresh portfolio with given starting cash."""
    return Portfolio(
        cash=float(initial_cash),
        margin_requirement=float(margin_requirement),
    )
