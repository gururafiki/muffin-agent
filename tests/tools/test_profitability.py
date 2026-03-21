"""Unit tests for profitability tools (ROIC, FCF conversion, accruals, CAGR)."""

import pytest

from muffin_agent.tools.profitability import (
    compute_accruals_ratio,
    compute_fcf_conversion,
    compute_revenue_cagr,
    compute_roic,
)


@pytest.mark.unit
class TestComputeROICTool:
    def test_basic(self):
        result = compute_roic.invoke(
            {"ebit": 10, "tax_rate": 0.21, "equity": 50, "debt": 20, "cash": 5}
        )
        assert result == pytest.approx(12.153846, rel=1e-3)

    def test_zero_invested_capital(self):
        result = compute_roic.invoke(
            {"ebit": 10, "tax_rate": 0.21, "equity": 0, "debt": 0, "cash": 10}
        )
        assert result is None


@pytest.mark.unit
class TestFCFConversionTool:
    def test_basic(self):
        result = compute_fcf_conversion.invoke({"fcf": 80, "net_income": 100})
        assert result == pytest.approx(80.0)

    def test_negative_net_income(self):
        result = compute_fcf_conversion.invoke({"fcf": 80, "net_income": -10})
        assert result is None


@pytest.mark.unit
class TestAccrualsRatioTool:
    def test_basic(self):
        result = compute_accruals_ratio.invoke(
            {
                "net_income": 100,
                "fcf": 80,
                "total_assets_current": 500,
                "total_assets_prior": 450,
            }
        )
        # (100 - 80) / ((500 + 450)/2) = 20/475 ≈ 0.0421
        assert result == pytest.approx(0.0421, rel=0.01)

    def test_no_prior(self):
        result = compute_accruals_ratio.invoke(
            {"net_income": 100, "fcf": 80, "total_assets_current": 500}
        )
        assert result == pytest.approx(0.04, rel=0.01)


@pytest.mark.unit
class TestRevenueCAGRTool:
    def test_basic(self):
        result = compute_revenue_cagr.invoke(
            {"revenue_start": 100, "revenue_end": 133.1, "years": 3}
        )
        assert result == pytest.approx(10.0, rel=0.01)
