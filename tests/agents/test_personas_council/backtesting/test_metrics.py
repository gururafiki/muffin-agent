"""Unit tests for backtest performance metrics."""

from __future__ import annotations

import pytest

from muffin_agent.agents.personas_council.backtesting.metrics import (
    compute_benchmark_comparison,
    compute_max_drawdown,
    compute_returns_from_equity,
    compute_sharpe,
    compute_sortino,
    compute_total_return,
)


@pytest.mark.unit
class TestComputeSharpe:
    def test_steady_returns(self):
        # Constant positive returns → very high Sharpe
        s = compute_sharpe([0.01] * 12, frequency=12)
        # zero variance — should return None
        assert s is None

    def test_variable_returns(self):
        s = compute_sharpe([0.01, 0.02, 0.005, 0.015, 0.012], frequency=12)
        assert s is not None
        assert s > 0  # net-positive excess returns

    def test_negative_returns(self):
        s = compute_sharpe([-0.02, -0.01, -0.03, -0.005, -0.015], frequency=12)
        assert s is not None
        assert s < 0

    def test_insufficient_data(self):
        assert compute_sharpe([0.01], frequency=12) is None
        assert compute_sharpe([], frequency=12) is None


@pytest.mark.unit
class TestComputeSortino:
    def test_returns_positive_for_upside_skew(self):
        s = compute_sortino([0.05, 0.03, -0.01, 0.04, 0.02], frequency=12)
        assert s is not None
        assert s > 0

    def test_no_downside_returns_none(self):
        # All returns above risk-free → no downside → None
        s = compute_sortino([0.10] * 6, risk_free_rate=0.0, frequency=12)
        assert s is None

    def test_insufficient_data(self):
        assert compute_sortino([0.01], frequency=12) is None


@pytest.mark.unit
class TestComputeMaxDrawdown:
    def test_no_drawdown(self):
        assert compute_max_drawdown([100, 110, 120, 130]) == 0.0

    def test_simple_drawdown(self):
        dd = compute_max_drawdown([100, 120, 90, 110])
        # Peak 120 → trough 90 = -25% drawdown
        assert dd == pytest.approx(-0.25)

    def test_recovers_then_deeper(self):
        dd = compute_max_drawdown([100, 110, 105, 130, 90, 95])
        # Peak 130 → trough 90 = -30.77% drawdown (deeper than first dip)
        assert dd == pytest.approx(-40 / 130, abs=1e-6)

    def test_insufficient_data(self):
        assert compute_max_drawdown([100]) is None
        assert compute_max_drawdown([]) is None


@pytest.mark.unit
class TestComputeTotalReturn:
    def test_simple(self):
        # 100 → 150 = +50%
        assert compute_total_return([100, 110, 130, 150]) == pytest.approx(0.5)

    def test_negative(self):
        assert compute_total_return([100, 80]) == pytest.approx(-0.20)

    def test_short(self):
        assert compute_total_return([100]) is None
        assert compute_total_return([0, 100]) is None


@pytest.mark.unit
class TestComputeReturnsFromEquity:
    def test_basic(self):
        ret = compute_returns_from_equity([100, 110, 121])
        assert ret == [pytest.approx(0.10), pytest.approx(0.10)]

    def test_skips_zero_prev(self):
        ret = compute_returns_from_equity([0, 100, 110])
        # The first prev=0 entry is skipped
        assert len(ret) == 1
        assert ret[0] == pytest.approx(0.10)

    def test_empty(self):
        assert compute_returns_from_equity([]) == []


@pytest.mark.unit
class TestBenchmarkComparison:
    def test_alpha_when_portfolio_outperforms(self):
        result = compute_benchmark_comparison(
            portfolio_curve=[100, 110, 125],
            benchmark_prices=[100, 105, 110],
        )
        assert result["portfolio_total_return"] == pytest.approx(0.25)
        assert result["benchmark_total_return"] == pytest.approx(0.10)
        assert result["alpha"] == pytest.approx(0.15)
        assert result["tracking_error"] is not None

    def test_alpha_negative_when_underperforms(self):
        result = compute_benchmark_comparison(
            portfolio_curve=[100, 95, 90],
            benchmark_prices=[100, 105, 110],
        )
        assert result["alpha"] is not None
        assert result["alpha"] < 0

    def test_handles_missing_benchmark(self):
        result = compute_benchmark_comparison(
            portfolio_curve=[100, 110],
            benchmark_prices=[],
        )
        # alpha computation requires both legs, tracking_error needs 2+ paired
        assert result["benchmark_total_return"] is None
        assert result["alpha"] is None
        assert result["tracking_error"] is None
