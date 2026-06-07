"""Unit tests for the trade executor."""

from __future__ import annotations

import pytest

from muffin_agent.agents.personas_council.portfolio.executor import (
    PortfolioOrder,
    apply_orders,
)
from muffin_agent.agents.personas_council.portfolio.state import new_portfolio


@pytest.mark.unit
class TestApplyOrders:
    def test_buy_then_sell_round_trip(self):
        p = new_portfolio(initial_cash=100_000)
        orders = [
            PortfolioOrder(ticker="AAPL", action="buy", quantity=100),
            PortfolioOrder(ticker="AAPL", action="sell", quantity=100),
        ]
        new_p, trades = apply_orders(p, orders, {"AAPL": 150})
        assert len(trades) == 2
        assert all(t.executed_quantity == 100 for t in trades)
        # Bought + sold at same price = no net change
        assert new_p.cash == 100_000
        assert new_p.positions["AAPL"].long == 0

    def test_hold_passes_through(self):
        p = new_portfolio(initial_cash=100_000)
        orders = [PortfolioOrder(ticker="AAPL", action="hold", quantity=0)]
        new_p, trades = apply_orders(p, orders, {"AAPL": 150})
        assert len(trades) == 1
        assert trades[0].action == "hold"
        assert trades[0].executed_quantity == 0
        assert not trades[0].skipped
        assert new_p is p

    def test_missing_price_marks_skipped(self):
        p = new_portfolio(initial_cash=100_000)
        orders = [PortfolioOrder(ticker="AAPL", action="buy", quantity=10)]
        new_p, trades = apply_orders(p, orders, {})
        assert trades[0].skipped
        assert "No price" in trades[0].reason
        assert new_p is p

    def test_sell_with_no_position_skipped(self):
        p = new_portfolio(initial_cash=100_000)
        orders = [PortfolioOrder(ticker="AAPL", action="sell", quantity=10)]
        new_p, trades = apply_orders(p, orders, {"AAPL": 150})
        assert trades[0].skipped
        assert "Insufficient" in trades[0].reason
        # Cash unchanged
        assert new_p.cash == 100_000

    def test_short_then_cover(self):
        p = new_portfolio(initial_cash=100_000, margin_requirement=0.5)
        orders = [
            PortfolioOrder(ticker="AAPL", action="short", quantity=50),
            PortfolioOrder(ticker="AAPL", action="cover", quantity=50),
        ]
        new_p, trades = apply_orders(p, orders, {"AAPL": 300})
        assert len(trades) == 2
        # Short cover at same price = no realised gain, cash unchanged
        assert new_p.cash == 100_000
        assert new_p.margin_used == 0

    def test_executed_log_carries_partial_fill(self):
        # Cash only allows 6 shares at $300 with 50% margin
        p = new_portfolio(initial_cash=1_000, margin_requirement=0.5)
        orders = [PortfolioOrder(ticker="AAPL", action="short", quantity=100)]
        new_p, trades = apply_orders(p, orders, {"AAPL": 300})
        assert trades[0].executed_quantity == 6
        assert trades[0].requested_quantity == 100
        assert not trades[0].skipped
