"""Unit tests for portfolio state + pure mutation helpers."""

from __future__ import annotations

import pytest

from muffin_agent.agents.personas_council.portfolio.state import (
    Portfolio,
    apply_long_buy,
    apply_long_sell,
    apply_short_cover,
    apply_short_open,
    mark_to_market,
    new_portfolio,
)


@pytest.mark.unit
class TestNewPortfolio:
    def test_starts_with_initial_cash(self):
        p = new_portfolio(initial_cash=100_000)
        assert p.cash == 100_000
        assert p.margin_used == 0
        assert p.margin_requirement == 0.5
        assert p.positions == {}
        assert p.realized_gains == {}


@pytest.mark.unit
class TestApplyLongBuy:
    def test_full_buy(self):
        p = new_portfolio(initial_cash=100_000)
        p2, qty = apply_long_buy(p, "AAPL", 100, 150)
        assert qty == 100
        assert p2.cash == 100_000 - 15_000
        assert p2.positions["AAPL"].long == 100
        assert p2.positions["AAPL"].long_cost_basis == 150
        # Original portfolio not mutated
        assert p.cash == 100_000

    def test_partial_buy_when_cash_short(self):
        p = new_portfolio(initial_cash=10_000)
        p2, qty = apply_long_buy(p, "AAPL", 100, 150)
        # Can only afford 66 shares: 66*150=9900 ≤ 10000
        assert qty == 66
        assert p2.cash == 10_000 - 66 * 150

    def test_zero_quantity(self):
        p = new_portfolio(initial_cash=10_000)
        p2, qty = apply_long_buy(p, "AAPL", 0, 150)
        assert qty == 0
        assert p2 is p

    def test_weighted_avg_cost_basis(self):
        p = new_portfolio(initial_cash=100_000)
        p2, _ = apply_long_buy(p, "AAPL", 50, 100)
        p3, _ = apply_long_buy(p2, "AAPL", 50, 120)
        # avg cost = (50*100 + 50*120) / 100 = 110
        assert p3.positions["AAPL"].long == 100
        assert p3.positions["AAPL"].long_cost_basis == pytest.approx(110)


@pytest.mark.unit
class TestApplyLongSell:
    def test_full_sell_realises_gain(self):
        p = new_portfolio(initial_cash=100_000)
        p2, _ = apply_long_buy(p, "AAPL", 100, 150)
        p3, qty = apply_long_sell(p2, "AAPL", 100, 160)
        assert qty == 100
        assert p3.cash == 100_000 + 1_000  # bought 15k → sold 16k
        assert p3.realized_gains["AAPL"].long == 1_000
        assert p3.positions["AAPL"].long == 0
        assert p3.positions["AAPL"].long_cost_basis == 0

    def test_sell_more_than_owned_caps(self):
        p = new_portfolio(initial_cash=100_000)
        p2, _ = apply_long_buy(p, "AAPL", 50, 100)
        p3, qty = apply_long_sell(p2, "AAPL", 100, 110)
        assert qty == 50

    def test_sell_with_no_position(self):
        p = new_portfolio(initial_cash=100_000)
        p2, qty = apply_long_sell(p, "AAPL", 10, 150)
        assert qty == 0
        assert p2 is p


@pytest.mark.unit
class TestShortRoundTrip:
    def test_short_open_uses_margin(self):
        p = new_portfolio(initial_cash=100_000, margin_requirement=0.5)
        p2, qty = apply_short_open(p, "AAPL", 50, 300)
        # proceeds 15k, margin required = 7.5k, cash net = 100k + 15k - 7.5k = 107.5k
        assert qty == 50
        assert p2.cash == 107_500
        assert p2.margin_used == 7_500
        assert p2.positions["AAPL"].short == 50
        assert p2.positions["AAPL"].short_cost_basis == 300

    def test_short_cover_realises_gain(self):
        p = new_portfolio(initial_cash=100_000, margin_requirement=0.5)
        p2, _ = apply_short_open(p, "AAPL", 50, 300)
        p3, qty = apply_short_cover(p2, "AAPL", 50, 280)
        assert qty == 50
        # released margin 7.5k, cover cost 14k → cash back to 100k + 1k profit
        assert p3.cash == 101_000
        assert p3.margin_used == 0
        assert p3.realized_gains["AAPL"].short == 1_000
        assert p3.positions["AAPL"].short == 0
        assert p3.positions["AAPL"].short_cost_basis == 0
        assert p3.positions["AAPL"].short_margin_used == 0

    def test_short_partial_when_margin_short(self):
        p = new_portfolio(initial_cash=1_000, margin_requirement=0.5)
        # Can short at most 6 shares at $300: 6*300*0.5 = 900 margin ≤ 1000 cash
        p2, qty = apply_short_open(p, "AAPL", 100, 300)
        assert qty == 6


@pytest.mark.unit
class TestMarkToMarket:
    def test_nav_after_round_trip(self):
        p = new_portfolio(initial_cash=100_000)
        p2, _ = apply_long_buy(p, "AAPL", 100, 150)
        p3, _ = apply_long_sell(p2, "AAPL", 100, 160)
        val = mark_to_market(p3, {"AAPL": 160})
        # cash = 101000, no positions remaining, NAV = cash
        assert val.nav == 101_000
        assert val.long_value == 0
        assert val.short_exposure == 0
        assert val.realized_gains_total == 1_000

    def test_long_value_at_current_price(self):
        p = new_portfolio(initial_cash=100_000)
        p2, _ = apply_long_buy(p, "AAPL", 100, 150)
        val = mark_to_market(p2, {"AAPL": 160})
        # cash 85k + long_value 16k = NAV 101k (unrealised gain)
        assert val.long_value == 16_000
        assert val.nav == 101_000

    def test_missing_price_skipped(self):
        p = new_portfolio(initial_cash=100_000)
        p2, _ = apply_long_buy(p, "AAPL", 100, 150)
        val = mark_to_market(p2, {})  # no prices
        # No long_value contribution → NAV = cash only
        assert val.long_value == 0
        assert val.nav == 85_000


@pytest.mark.unit
class TestSerialization:
    def test_pydantic_round_trip(self):
        p = new_portfolio(initial_cash=100_000)
        p2, _ = apply_long_buy(p, "AAPL", 100, 150)
        dump = p2.model_dump()
        rebuilt = Portfolio.model_validate(dump)
        assert rebuilt.cash == p2.cash
        assert rebuilt.positions["AAPL"].long == 100
        assert rebuilt.positions["AAPL"].long_cost_basis == 150
