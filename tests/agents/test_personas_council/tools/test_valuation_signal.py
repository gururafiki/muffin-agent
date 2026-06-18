"""Tests for the deterministic multi-method valuation scoring."""

from __future__ import annotations

import pytest

from muffin_agent.agents.personas_council.tools.valuation_signal import (
    calculate_owner_earnings_value,
    calculate_wacc,
    score_valuation_signals,
)


@pytest.mark.unit
class TestHelpers:
    def test_owner_earnings_positive(self):
        # NI 100 + D&A 20 - capex 30 - ΔWC 5 = 85 owner earnings → positive PV
        val = calculate_owner_earnings_value(100, 20, 30, 5, growth_rate=0.05)
        assert val > 0

    def test_owner_earnings_non_positive_returns_zero(self):
        assert calculate_owner_earnings_value(10, 5, 50, 0) == 0.0

    def test_owner_earnings_missing_returns_zero(self):
        assert calculate_owner_earnings_value(None, 20, 30, 5) == 0.0

    def test_wacc_bounded(self):
        w = calculate_wacc(1000, 200, 50, 8.0, 0.4)
        assert 0.06 <= w <= 0.20


@pytest.mark.unit
class TestCombine:
    def test_undervalued_is_bullish(self):
        # Big FCF + small market cap → large positive gap
        result = score_valuation_signals(
            market_cap=100.0,
            net_income=50.0,
            depreciation=10.0,
            capital_expenditure=5.0,
            working_capital_change=0.0,
            earnings_growth=0.10,
            revenue_growth=0.12,
            free_cash_flow_history=[40.0, 45.0, 50.0],
            total_debt=20.0,
            cash=30.0,
            interest_coverage=10.0,
            debt_to_equity=0.3,
            enterprise_value=120.0,
            ev_to_ebitda_history=[8.0, 9.0, 7.0],
            price_to_book_ratio=2.0,
            book_value_growth=0.05,
        )
        assert result["signal"] == "bullish"
        assert result["weighted_gap"] > 0.15

    def test_overvalued_is_bearish(self):
        result = score_valuation_signals(
            market_cap=10_000.0,
            net_income=50.0,
            depreciation=10.0,
            capital_expenditure=40.0,
            working_capital_change=20.0,
            earnings_growth=0.02,
            revenue_growth=0.02,
            free_cash_flow_history=[20.0, 18.0, 15.0],
            total_debt=500.0,
            cash=10.0,
            interest_coverage=2.0,
            debt_to_equity=2.0,
            enterprise_value=10_500.0,
            ev_to_ebitda_history=[40.0, 45.0, 50.0],
            price_to_book_ratio=20.0,
            book_value_growth=0.0,
        )
        assert result["signal"] == "bearish"

    def test_missing_market_cap_is_neutral(self):
        result = score_valuation_signals(
            market_cap=None,
            net_income=50.0,
            depreciation=10.0,
            capital_expenditure=5.0,
            working_capital_change=0.0,
            earnings_growth=0.1,
            revenue_growth=0.1,
            free_cash_flow_history=[40.0],
            total_debt=0.0,
            cash=0.0,
            interest_coverage=None,
            debt_to_equity=None,
            enterprise_value=None,
            ev_to_ebitda_history=None,
            price_to_book_ratio=None,
            book_value_growth=None,
        )
        assert result["signal"] == "neutral"
        assert result["confidence"] == 0.0

    def test_method_weights(self):
        result = score_valuation_signals(
            market_cap=1000.0,
            net_income=120.0,
            depreciation=30.0,
            capital_expenditure=25.0,
            working_capital_change=5.0,
            earnings_growth=0.10,
            revenue_growth=0.12,
            free_cash_flow_history=[80.0, 90.0, 110.0],
            total_debt=200.0,
            cash=150.0,
            interest_coverage=8.0,
            debt_to_equity=0.4,
            enterprise_value=1050.0,
            ev_to_ebitda_history=[10.0, 11.0, 9.0],
            price_to_book_ratio=4.0,
            book_value_growth=0.05,
        )
        assert result["methods"]["dcf"]["weight"] == 0.35
        assert result["methods"]["owner_earnings"]["weight"] == 0.35
        assert result["methods"]["ev_ebitda"]["weight"] == 0.20
        assert result["methods"]["residual_income"]["weight"] == 0.10
