"""Unit tests for projection tools (3-year financials, sensitivity)."""

import pytest

from muffin_agent.tools.projections import (
    compute_sensitivity,
    project_three_year_financials,
)


@pytest.mark.unit
class TestSensitivityTool:
    def test_basic(self):
        result = compute_sensitivity.invoke(
            {
                "baseline_revenue": 1e9,
                "ebit_margin": 0.15,
                "tax_rate": 0.21,
                "diluted_shares": 1e8,
                "capex": 1e8,
            }
        )
        assert result["delta_eps_per_rev_1pp"] is not None
        assert result["delta_eps_per_margin_1pp"] is not None
        assert result["delta_fcf_per_capex_10pct"] == pytest.approx(1e7)

    def test_output_schema_in_extras(self):
        schema = compute_sensitivity.extras["output_schema"]
        assert "properties" in schema
        assert "delta_eps_per_rev_1pp" in schema["properties"]


@pytest.mark.unit
class TestProjectThreeYearFinancialsTool:
    def test_basic(self):
        result = project_three_year_financials.invoke(
            {
                "baseline_revenue": 100.0,
                "base_year": 2024,
                "tax_rate": 0.21,
                "da_pct_rev": 0.05,
                "rev_growth_y1": 0.10,
                "rev_growth_y2": 0.08,
                "rev_growth_y3": 0.06,
                "ebitda_margin_y1": 0.30,
                "ebitda_margin_y2": 0.31,
                "ebitda_margin_y3": 0.32,
                "capex_rate_y1": 0.08,
                "capex_rate_y2": 0.07,
                "capex_rate_y3": 0.07,
                "total_debt_0": 50.0,
                "cash_0": 20.0,
                "equity_0": 80.0,
                "fixed_assets_0": 60.0,
                "nwc_pct_rev": 0.12,
                "diluted_shares": 10.0,
            }
        )
        assert len(result) == 3
        assert result[0]["year"] == 2025
        assert result[1]["year"] == 2026
        assert result[2]["year"] == 2027
        # Verify income fields
        assert result[0]["revenue"] == pytest.approx(110.0)
        assert result[0]["ebitda"] == pytest.approx(33.0)
        assert result[0]["eps"] is not None
        # Verify balance sheet fields
        assert "total_debt" in result[0]
        assert "cash" in result[0]
        assert "shareholders_equity" in result[0]
        assert "total_assets" in result[0]

    def test_no_shares(self):
        result = project_three_year_financials.invoke(
            {
                "baseline_revenue": 100.0,
                "base_year": 2024,
                "tax_rate": 0.21,
                "da_pct_rev": 0.05,
                "rev_growth_y1": 0.10,
                "rev_growth_y2": 0.08,
                "rev_growth_y3": 0.06,
                "ebitda_margin_y1": 0.30,
                "ebitda_margin_y2": 0.30,
                "ebitda_margin_y3": 0.30,
                "capex_rate_y1": 0.08,
                "capex_rate_y2": 0.08,
                "capex_rate_y3": 0.08,
                "total_debt_0": 50.0,
                "cash_0": 20.0,
                "equity_0": 80.0,
                "fixed_assets_0": 60.0,
                "nwc_pct_rev": 0.12,
            }
        )
        assert result[0]["eps"] is None

    def test_with_dividends_buybacks(self):
        result = project_three_year_financials.invoke(
            {
                "baseline_revenue": 100.0,
                "base_year": 2024,
                "tax_rate": 0.21,
                "da_pct_rev": 0.05,
                "rev_growth_y1": 0.10,
                "rev_growth_y2": 0.10,
                "rev_growth_y3": 0.10,
                "ebitda_margin_y1": 0.30,
                "ebitda_margin_y2": 0.30,
                "ebitda_margin_y3": 0.30,
                "capex_rate_y1": 0.08,
                "capex_rate_y2": 0.08,
                "capex_rate_y3": 0.08,
                "total_debt_0": 50.0,
                "cash_0": 20.0,
                "equity_0": 80.0,
                "fixed_assets_0": 60.0,
                "nwc_pct_rev": 0.12,
                "diluted_shares": 10.0,
                "dividends": 5.0,
                "buybacks": 3.0,
            }
        )
        # Dividends and buybacks should reduce cash and equity
        # equity + net_income - divs - buybacks
        assert result[0]["shareholders_equity"] < 80.0 + 20

    def test_output_schema_in_extras(self):
        schema = project_three_year_financials.extras["output_schema"]
        assert "properties" in schema
        assert "year" in schema["properties"]
        assert "revenue" in schema["properties"]
