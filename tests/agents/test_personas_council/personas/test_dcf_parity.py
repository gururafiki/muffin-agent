"""Persona valuation-parity + metrics-ordering tests."""

from __future__ import annotations

import pytest

from muffin_agent.agents.personas_council.personas import aswath_damodaran as dam
from muffin_agent.agents.personas_council.personas import bill_ackman as ack
from muffin_agent.agents.personas_council.personas import rakesh_jhunjhunwala as rak
from muffin_agent.agents.personas_council.personas.warren_buffett import (
    BuffettMetricsRow,
    compute_evidence_node,
)


@pytest.mark.unit
class TestBuffettOrdering:
    """metrics_history is now oldest → newest (latest = last element)."""

    def test_latest_snapshot_is_last(self):
        state = {
            "ticker": "X",
            # oldest weak → newest strong
            "metrics_history": [
                BuffettMetricsRow(return_on_equity=0.05, operating_margin=0.05),
                BuffettMetricsRow(return_on_equity=0.10, operating_margin=0.10),
                BuffettMetricsRow(return_on_equity=0.30, operating_margin=0.30),
            ],
        }
        ev = compute_evidence_node(state)["evidence"]
        # Fundamentals must read the NEWEST (last) row → ROE 0.30
        assert ev.fundamentals.roe_value == pytest.approx(0.30)


@pytest.mark.unit
class TestBuffettDerivedGrowth:
    def test_owner_earnings_and_intrinsic_present(self):
        state = {
            "ticker": "X",
            "metrics_history": [
                BuffettMetricsRow(
                    return_on_equity=0.25,
                    debt_to_equity=0.4,
                    operating_margin=0.28,
                    current_ratio=1.5,
                    asset_turnover=1.1,
                )
                for _ in range(5)
            ],
            "net_income_series": [60, 70, 80, 90, 100],
            "revenue_series": [300, 340, 380, 440, 500],
            "depreciation_amortization_series": [18, 18, 19, 19, 20],
            "capital_expenditure_series": [30, 32, 35, 38, 40],
            "current_assets_series": [150, 160, 170, 180, 200],
            "current_liabilities_series": [80, 85, 90, 95, 100],
            "market_cap": 1000,
        }
        ev = compute_evidence_node(state)["evidence"]
        assert ev.owner_earnings is not None
        assert ev.intrinsic_value is not None
        assert ev.margin_of_safety_pct is not None


@pytest.mark.unit
class TestAckmanActivismAndValuation:
    def test_activism_revenue_growth_low_margin(self):
        state = {
            "ticker": "X",
            "revenue_series": [100, 120, 150],  # +50%
            "operating_margin_series": [0.06, 0.07, 0.08],  # avg 7% < 10%
            "free_cash_flow_series": [10, 12, 15],
            "total_debt_series": [30, 30, 30],
            "shareholders_equity_series": [80, 90, 100],
            "dividends_series": [-5, -5, -5],
            "outstanding_shares_series": [20, 20, 20],
            "market_cap": 80,
        }
        ev = ack.compute_evidence_node(state)["evidence"]
        assert ev.activism_potential.score == 2
        assert ev.activism_potential.revenue_growth_pct == pytest.approx(50.0)
        assert ev.activism_potential.avg_operating_margin == pytest.approx(0.07)
        # Exit-multiple DCF on positive FCF yields an intrinsic value
        assert ev.intrinsic_value is not None
        assert ev.valuation.max_score == 3


@pytest.mark.unit
class TestRakeshTieredDCF:
    def test_high_growth_capped_and_intrinsic_present(self):
        state = {
            "ticker": "X",
            "revenue_series": [100, 120, 150, 180, 220],
            "net_income_series": [10, 13, 17, 22, 28],  # ~29% CAGR → capped 20%
            "eps_series": [1.0, 1.3, 1.7, 2.2, 2.8],
            "free_cash_flow_series": [8, 10, 14, 18, 24],
            "dividends_series": [-1, -1, -1, -1, -1],
            "issuance_or_purchase_series": [0, 0, 0, 0, -5],
            "total_assets_series": [200, 220, 250, 290, 340],
            "total_liabilities_series": [80, 85, 90, 95, 100],
            "current_assets_series": [120, 130, 140, 160, 190],
            "current_liabilities_series": [60, 62, 65, 70, 75],
            "roe_latest": 0.24,
            "operating_margin_latest": 0.22,
            "market_cap": 300,
        }
        ev = rak.compute_evidence_node(state)["evidence"]
        assert ev.intrinsic_value is not None
        assert ev.quality_tier in ("high", "medium", "low")
        assert ev.discount_rate in (0.12, 0.15, 0.18)


@pytest.mark.unit
class TestDamodaranRelativePE:
    def test_cheap_vs_median(self):
        state = {
            "ticker": "X",
            "revenue_series": [100, 110, 121, 133, 146],
            "free_cash_flow_series": [10, 11, 13, 15, 18],
            "roic_latest": 0.12,
            "beta_latest": 1.0,
            "debt_to_equity_latest": 0.5,
            "interest_coverage_latest": 8.0,
            "pe_ratio_history": [22, 21, 20, 18, 10],  # latest 10 < 0.7*median(20)
            "market_cap": 150,
        }
        ev = dam.compute_evidence_node(state)["evidence"]
        assert ev.relative_valuation.score == 1

    def test_expensive_vs_median(self):
        state = {
            "ticker": "X",
            "revenue_series": [100, 110, 121, 133, 146],
            "free_cash_flow_series": [10, 11, 13, 15, 18],
            "beta_latest": 1.0,
            "pe_ratio_history": [18, 19, 20, 21, 30],  # latest 30 > 1.3*median(20)
            "market_cap": 150,
        }
        ev = dam.compute_evidence_node(state)["evidence"]
        assert ev.relative_valuation.score == -1

    def test_insufficient_history_is_zero(self):
        state = {"ticker": "X", "pe_ratio_history": [20, 18]}
        ev = dam.compute_evidence_node(state)["evidence"]
        assert ev.relative_valuation.score == 0
