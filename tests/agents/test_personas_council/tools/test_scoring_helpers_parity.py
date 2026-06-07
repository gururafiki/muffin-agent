"""Parity tests for the DCF helpers restored to ai-hedge-fund behaviour (W1)."""

from __future__ import annotations

import pytest

from muffin_agent.agents.personas_council.tools.scoring_helpers import (
    compute_buffett_3stage_dcf,
    compute_buffett_owner_earnings,
    compute_damodaran_fcff_dcf,
    compute_intrinsic_value_exit_multiple,
    estimate_maintenance_capex,
)


@pytest.mark.unit
class TestMaintenanceCapex:
    def test_median_of_three_when_enough_ratios(self):
        # capex 40 → m1=34; dep 20 → m2=20; ratios≈avg×rev → m3≈42; median=34
        mc = estimate_maintenance_capex(
            [35, 38, 40], [18, 19, 20], [400, 450, 500]
        )
        assert mc == pytest.approx(34.0, abs=0.5)

    def test_max_of_two_when_few_ratios(self):
        # only 1 valid ratio period → max(0.85*capex, depreciation)
        mc = estimate_maintenance_capex([40], [25], [500])
        assert mc == pytest.approx(max(40 * 0.85, 25))


@pytest.mark.unit
class TestBuffettOwnerEarnings:
    def test_full_formula_with_working_capital(self):
        # NI 100 + D&A 20 - maint_capex - ΔWC
        # maint_capex: capex 40 → m1=34, dep 20, ratios → median ≈ 34
        # ΔWC = (200-120) - (180-110) = 80 - 70 = 10
        oe = compute_buffett_owner_earnings(
            net_income_series=[80, 90, 100],
            depreciation_series=[18, 19, 20],
            capex_series=[35, 38, 40],
            revenue_series=[400, 450, 500],
            current_assets_series=[160, 180, 200],
            current_liabilities_series=[100, 110, 120],
        )
        assert oe == pytest.approx(100 + 20 - 34 - 10, abs=0.6)

    def test_zero_working_capital_when_history_missing(self):
        oe = compute_buffett_owner_earnings(
            [100], [20], [40], [500]
        )
        # ΔWC defaults to 0 → 100 + 20 - max(34,20) = 86
        assert oe == pytest.approx(86.0, abs=0.5)

    def test_missing_latest_returns_none(self):
        assert (
            compute_buffett_owner_earnings([None], [20], [40], [500]) is None
        )


@pytest.mark.unit
class TestBuffett3StageGrowth:
    def test_explicit_stage2_used(self):
        with_explicit = compute_buffett_3stage_dcf(
            100, growth_stage_1=0.08, growth_stage_2=0.02
        )
        default_stage2 = compute_buffett_3stage_dcf(100, growth_stage_1=0.08)
        # default stage2 = 0.08/2 = 0.04 > 0.02 → higher IV than the explicit 0.02
        assert default_stage2 > with_explicit

    def test_lower_growth_lower_value(self):
        slow = compute_buffett_3stage_dcf(100, growth_stage_1=0.02, growth_stage_2=0.01)
        fast = compute_buffett_3stage_dcf(100, growth_stage_1=0.08, growth_stage_2=0.04)
        assert fast > slow  # history-derived growth materially changes IV


@pytest.mark.unit
class TestDamodaranTerminalBasis:
    def test_base_fcff_default_understates_vs_final_cf(self):
        base = compute_damodaran_fcff_dcf(100, 0.10, 1.0)
        final = compute_damodaran_fcff_dcf(100, 0.10, 1.0, terminal_basis="final_cf")
        assert base is not None and final is not None
        # Upstream-parity (base_fcff) terminal anchors on the un-grown FCFF →
        # strictly lower intrinsic value than the textbook final-CF terminal.
        assert base[0] < final[0]

    def test_returns_discount_rate_from_capm(self):
        result = compute_damodaran_fcff_dcf(100, 0.10, 1.0)
        assert result is not None
        # r = rf 0.04 + beta 1.0 * ERP 0.05 = 0.09
        assert result[1] == pytest.approx(0.09)


@pytest.mark.unit
class TestExitMultiple:
    def test_matches_hand_calc(self):
        # Ackman: fcf 100, g 6%, disc 10%, 15x, 5y
        iv = compute_intrinsic_value_exit_multiple(100, 0.06, 0.10, 15.0, 5)
        assert iv is not None
        # terminal = 100*1.06^5*15 / 1.1^5
        expected_terminal = 100 * 1.06**5 * 15 / 1.1**5
        assert iv > expected_terminal  # plus the discounted explicit-period flows
