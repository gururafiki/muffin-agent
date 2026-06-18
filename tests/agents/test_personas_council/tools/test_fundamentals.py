"""Tests for the deterministic fundamentals scoring (tools/fundamentals.py)."""

from __future__ import annotations

import pytest

from muffin_agent.agents.personas_council.tools.fundamentals import (
    score_financial_health,
    score_fundamentals,
    score_growth,
    score_price_ratios,
    score_profitability,
)


@pytest.mark.unit
class TestSubSignals:
    def test_profitability_all_strong_is_bullish(self):
        s = score_profitability(0.20, 0.25, 0.20)
        assert s["signal"] == "bullish"
        assert s["score"] == 3

    def test_profitability_none_met_is_bearish(self):
        s = score_profitability(0.05, 0.05, 0.05)
        assert s["signal"] == "bearish"
        assert s["score"] == 0

    def test_profitability_one_met_is_neutral(self):
        s = score_profitability(0.20, 0.05, 0.05)
        assert s["signal"] == "neutral"

    def test_growth_thresholds(self):
        assert score_growth(0.11, 0.11, 0.11)["signal"] == "bullish"
        assert score_growth(0.05, 0.05, 0.05)["signal"] == "bearish"

    def test_financial_health_fcf_conversion(self):
        # current>1.5, D/E<0.5, FCF/sh > 0.8*EPS → 3 → bullish
        s = score_financial_health(2.0, 0.3, 5.0, 4.0)
        assert s["score"] == 3
        assert s["signal"] == "bullish"

    def test_price_ratios_inverted_mapping(self):
        # All three rich → bearish (high multiples are a sell signal)
        assert score_price_ratios(30, 5, 8)["signal"] == "bearish"
        # All cheap → bullish
        assert score_price_ratios(10, 1, 1)["signal"] == "bullish"

    def test_missing_values_do_not_count(self):
        s = score_profitability(None, None, None)
        assert s["score"] == 0
        assert "n/a" in s["details"]


@pytest.mark.unit
class TestCombine:
    def test_majority_vote_bullish(self):
        result = score_fundamentals(
            {
                "return_on_equity": 0.25,
                "net_margin": 0.25,
                "operating_margin": 0.25,
                "revenue_growth": 0.20,
                "earnings_growth": 0.20,
                "book_value_growth": 0.15,
                "current_ratio": 2.0,
                "debt_to_equity": 0.3,
                "free_cash_flow_per_share": 5.0,
                "earnings_per_share": 4.0,
                "price_to_earnings_ratio": 12,
                "price_to_book_ratio": 2,
                "price_to_sales_ratio": 1,
            }
        )
        assert result["signal"] == "bullish"
        # 4/4 agree → confidence 1.0
        assert result["confidence"] == pytest.approx(1.0)

    def test_tie_is_neutral(self):
        # profitability bullish, price_ratios bearish, growth+health neutral → tie
        result = score_fundamentals(
            {
                "return_on_equity": 0.30,
                "net_margin": 0.30,
                "operating_margin": 0.30,
                "revenue_growth": 0.05,
                "earnings_growth": 0.12,
                "book_value_growth": 0.05,
                "current_ratio": 1.0,
                "debt_to_equity": 1.5,
                "free_cash_flow_per_share": 6.0,
                "earnings_per_share": 6.0,
                "price_to_earnings_ratio": 30,
                "price_to_book_ratio": 45,
                "price_to_sales_ratio": 8,
            }
        )
        assert result["signal"] == "neutral"
        assert result["confidence"] == pytest.approx(0.25)
