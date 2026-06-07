"""Tests for the deterministic growth scoring (tools/growth.py)."""

from __future__ import annotations

import pytest

from muffin_agent.agents.personas_council.tools.growth import (
    score_growth_signals,
    score_insider_conviction,
    trend_slope,
)


@pytest.mark.unit
class TestTrendSlope:
    def test_increasing_series_positive_slope(self):
        assert trend_slope([0.05, 0.10, 0.15, 0.20]) > 0

    def test_decreasing_series_negative_slope(self):
        assert trend_slope([0.20, 0.15, 0.10, 0.05]) < 0

    def test_too_short_is_zero(self):
        assert trend_slope([0.1]) == 0.0
        assert trend_slope(None) == 0.0


@pytest.mark.unit
class TestInsiderConviction:
    def test_dollar_weighted_net_buying(self):
        s = score_insider_conviction(
            [
                {"transaction_shares": 1000, "transaction_value": 100_000},
                {"transaction_shares": -100, "transaction_value": 5_000},
            ]
        )
        assert s["net_flow_ratio"] > 0.5
        assert s["score"] == 1.0

    def test_falls_back_to_share_counts(self):
        s = score_insider_conviction([{"transaction_shares": -500}])
        assert s["net_flow_ratio"] == pytest.approx(-1.0)
        assert s["score"] == 0.2

    def test_no_trades_is_neutral(self):
        s = score_insider_conviction([])
        assert s["net_flow_ratio"] == 0.0
        assert s["score"] == 0.5


@pytest.mark.unit
class TestCombine:
    def test_strong_growth_is_bullish(self):
        result = score_growth_signals(
            revenue_growth=[0.10, 0.18, 0.25],
            eps_growth=[0.12, 0.20, 0.28],
            fcf_growth=[0.05, 0.12, 0.20],
            gross_margin=[0.45, 0.50, 0.55],
            operating_margin=[0.12, 0.15, 0.18],
            net_margin=[0.08, 0.10, 0.12],
            peg_ratio=0.8,
            price_to_sales_ratio=1.5,
            debt_to_equity=0.3,
            current_ratio=2.5,
            insider_trades=[{"transaction_shares": 1000, "transaction_value": 50_000}],
        )
        assert result["signal"] == "bullish"
        assert result["weighted_score"] > 0.6
        assert 0.0 <= result["confidence"] <= 1.0

    def test_weak_everything_is_bearish(self):
        result = score_growth_signals(
            revenue_growth=[0.02, 0.01, 0.00],
            eps_growth=[0.01, 0.00, -0.02],
            fcf_growth=[0.00, -0.05, -0.10],
            gross_margin=[0.30, 0.28, 0.25],
            operating_margin=[0.10, 0.08, 0.05],
            net_margin=[0.05, 0.03, 0.01],
            peg_ratio=4.0,
            price_to_sales_ratio=10.0,
            debt_to_equity=2.0,
            current_ratio=0.8,
            insider_trades=[{"transaction_shares": -1000, "transaction_value": 50_000}],
        )
        assert result["signal"] == "bearish"
        assert result["weighted_score"] < 0.4

    def test_weights_sum_to_one(self):
        from muffin_agent.agents.personas_council.tools.growth import _WEIGHTS

        assert sum(_WEIGHTS.values()) == pytest.approx(1.0)
