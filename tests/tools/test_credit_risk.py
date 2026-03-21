"""Unit tests for credit risk tools (net debt/EBITDA, interest coverage, Altman Z)."""

import pytest

from muffin_agent.tools.credit_risk import (
    compute_altman_z_score,
    compute_interest_coverage,
    compute_net_debt_to_ebitda,
)


@pytest.mark.unit
class TestNetDebtToEBITDATool:
    def test_basic(self):
        result = compute_net_debt_to_ebitda.invoke(
            {"debt": 100, "cash": 20, "ebitda": 40}
        )
        assert result == pytest.approx(2.0)


@pytest.mark.unit
class TestInterestCoverageTool:
    def test_basic(self):
        result = compute_interest_coverage.invoke({"ebit": 50, "interest_expense": 10})
        assert result == pytest.approx(5.0)


@pytest.mark.unit
class TestAltmanZScoreTool:
    def test_basic(self):
        result = compute_altman_z_score.invoke(
            {
                "working_capital": 50,
                "retained_earnings": 200,
                "ebit": 80,
                "market_cap": 500,
                "total_liabilities": 300,
                "total_assets": 600,
                "revenue": 400,
            }
        )
        assert result is not None
        assert result == pytest.approx(2.673, rel=0.01)  # grey zone
