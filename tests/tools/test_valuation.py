"""Unit tests for valuation tools (WACC, DCF, multiples value, scenario-weighted NAV)."""  # noqa: E501

import pytest

from muffin_agent.tools.valuation import (
    compute_dcf,
    compute_multiples_value,
    compute_scenario_weighted_value,
    compute_wacc,
)

# ── compute_wacc ──────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestComputeWacc:
    # beta=1.2, rf=4.5%, erp=5.5%, kd=4%, dw=0.30, ew=0.70, t=21%
    # ke = 0.045 + 1.2 * 0.055 = 0.045 + 0.066 = 0.111
    # kd_at = 0.04 * (1 - 0.21) = 0.0316
    # wacc = 0.70 * 0.111 + 0.30 * 0.0316 = 0.0777 + 0.00948 = 0.08718
    _BASE = dict(
        beta=1.2,
        risk_free_rate=0.045,
        equity_risk_premium=0.055,
        cost_of_debt=0.04,
        debt_weight=0.30,
        equity_weight=0.70,
        tax_rate=0.21,
    )

    def test_known_value_wacc(self):
        result = compute_wacc.invoke(self._BASE)
        assert result["wacc"] == pytest.approx(0.08718, abs=1e-5)

    def test_cost_of_equity_formula(self):
        result = compute_wacc.invoke(self._BASE)
        expected_ke = 0.045 + 1.2 * 0.055
        assert result["cost_of_equity"] == pytest.approx(expected_ke, abs=1e-6)

    def test_cost_of_debt_after_tax(self):
        result = compute_wacc.invoke(self._BASE)
        expected_kd = 0.04 * (1 - 0.21)
        assert result["cost_of_debt_after_tax"] == pytest.approx(expected_kd, abs=1e-6)

    def test_weights_not_summing_to_one_returns_none(self):
        bad = {**self._BASE, "debt_weight": 0.40, "equity_weight": 0.70}
        result = compute_wacc.invoke(bad)
        assert result["wacc"] is None
        assert result["cost_of_equity"] is None
        assert result["cost_of_debt_after_tax"] is None

    def test_weights_within_tolerance_accepted(self):
        # 0.301 + 0.700 = 1.001 — within ±0.01
        ok = {**self._BASE, "debt_weight": 0.301, "equity_weight": 0.700}
        result = compute_wacc.invoke(ok)
        assert result["wacc"] is not None

    def test_negative_beta_returns_none(self):
        negative_beta = {**self._BASE, "beta": -0.5}
        result = compute_wacc.invoke(negative_beta)
        assert result["wacc"] is None

    def test_zero_debt_weight(self):
        all_equity = {**self._BASE, "debt_weight": 0.0, "equity_weight": 1.0}
        result = compute_wacc.invoke(all_equity)
        ke = 0.045 + 1.2 * 0.055
        assert result["wacc"] == pytest.approx(ke, abs=1e-6)

    def test_output_schema_in_extras(self):
        schema = compute_wacc.extras["output_schema"]
        assert "properties" in schema
        for key in ("wacc", "cost_of_equity", "cost_of_debt_after_tax"):
            assert key in schema["properties"]


# ── compute_dcf ───────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestComputeDcf:
    # FCFs: 100, 110, 120 (millions); wacc=10%; shares=100M; net_debt=50
    # PV year1 = 100/1.10 = 90.909...
    # PV year2 = 110/1.21 = 90.909...
    # PV year3 = 120/1.331 = 90.157...
    # pv_fcfs = 271.975...
    _BASE = dict(
        fcf_year1=100.0,
        fcf_year2=110.0,
        fcf_year3=120.0,
        wacc=0.10,
        shares_outstanding=100.0,
        net_debt=50.0,
    )

    def test_pv_fcfs_known_value(self):
        result = compute_dcf.invoke({**self._BASE, "terminal_growth_rate": 0.025})
        expected_pv = (100 / 1.10) + (110 / 1.21) + (120 / 1.331)
        assert result["pv_fcfs"] == pytest.approx(expected_pv, rel=1e-4)

    def test_gordon_growth_terminal_value(self):
        # TV = 120 * 1.025 / (0.10 - 0.025) = 123 / 0.075 = 1640.0
        # PV_TV = 1640 / (1.10^3) = 1640 / 1.331 = 1232.156...
        result = compute_dcf.invoke({**self._BASE, "terminal_growth_rate": 0.025})
        expected_pv_tv = (120 * 1.025) / (0.10 - 0.025) / (1.10**3)
        assert result["pv_terminal_gordon_growth"] == pytest.approx(
            expected_pv_tv, rel=1e-4
        )

    def test_exit_multiple_terminal_value(self):
        # TV = 50 * 12 = 600; PV_TV = 600 / 1.331 = 450.789...
        result = compute_dcf.invoke(
            {**self._BASE, "terminal_year_ebitda": 50.0, "exit_ebitda_multiple": 12.0}
        )
        expected_pv_tv = (50.0 * 12.0) / (1.10**3)
        assert result["pv_terminal_exit_multiple"] == pytest.approx(
            expected_pv_tv, rel=1e-4
        )

    def test_nav_per_share_exit_only(self):
        result = compute_dcf.invoke(
            {**self._BASE, "terminal_year_ebitda": 50.0, "exit_ebitda_multiple": 12.0}
        )
        pv_fcfs = (100 / 1.10) + (110 / 1.21) + (120 / 1.331)
        pv_tv = (50.0 * 12.0) / (1.10**3)
        expected_nav = (pv_fcfs + pv_tv - 50.0) / 100.0
        assert result["nav_exit_multiple"] == pytest.approx(expected_nav, rel=1e-4)
        assert result["methodology"] == "exit_multiple"

    def test_nav_per_share_gordon_only(self):
        result = compute_dcf.invoke({**self._BASE, "terminal_growth_rate": 0.025})
        assert result["nav_gordon_growth"] is not None
        assert result["methodology"] == "gordon_growth"

    def test_blended_is_average_of_both(self):
        result = compute_dcf.invoke(
            {
                **self._BASE,
                "terminal_year_ebitda": 50.0,
                "exit_ebitda_multiple": 12.0,
                "terminal_growth_rate": 0.025,
            }
        )
        assert result["methodology"] == "blended"
        assert result["nav_exit_multiple"] is not None
        assert result["nav_gordon_growth"] is not None
        # Blending uses unrounded intermediates before the final round(4),
        # so compare with a small absolute tolerance.
        expected_blended = (
            result["nav_exit_multiple"] + result["nav_gordon_growth"]
        ) / 2.0
        assert result["nav_per_share"] == pytest.approx(expected_blended, abs=1e-3)

    def test_wacc_le_terminal_growth_disables_gordon_growth(self):
        result = compute_dcf.invoke(
            {**self._BASE, "terminal_growth_rate": 0.10}
        )  # wacc == terminal_growth_rate
        assert result["nav_gordon_growth"] is None
        assert result["pv_terminal_gordon_growth"] is None

    def test_net_cash_increases_nav(self):
        with_net_cash = {
            **self._BASE,
            "net_debt": -200.0,  # net cash
            "terminal_growth_rate": 0.025,
        }
        with_net_debt = {**self._BASE, "net_debt": 200.0, "terminal_growth_rate": 0.025}
        nav_cash = compute_dcf.invoke(with_net_cash)["nav_per_share"]
        nav_debt = compute_dcf.invoke(with_net_debt)["nav_per_share"]
        assert nav_cash > nav_debt

    def test_zero_wacc_returns_none(self):
        result = compute_dcf.invoke(
            {**self._BASE, "wacc": 0.0, "terminal_growth_rate": 0.025}
        )
        assert result["nav_per_share"] is None

    def test_zero_shares_returns_none(self):
        result = compute_dcf.invoke(
            {**self._BASE, "shares_outstanding": 0.0, "terminal_growth_rate": 0.025}
        )
        assert result["nav_per_share"] is None

    def test_no_terminal_method_returns_none_nav(self):
        result = compute_dcf.invoke(self._BASE)  # no terminal inputs
        assert result["nav_per_share"] is None
        assert result["methodology"] is None

    def test_output_schema_in_extras(self):
        schema = compute_dcf.extras["output_schema"]
        assert "properties" in schema
        for key in (
            "nav_per_share",
            "nav_exit_multiple",
            "nav_gordon_growth",
            "pv_fcfs",
            "methodology",
        ):
            assert key in schema["properties"]


# ── compute_multiples_value ───────────────────────────────────────────────────


@pytest.mark.unit
class TestComputeMultiplesValue:
    def test_ev_ebitda_known_value(self):
        # (200 * 12 - 100) / 50 = (2400 - 100) / 50 = 46.0
        result = compute_multiples_value.invoke(
            {
                "ntm_metric": 200.0,
                "multiple": 12.0,
                "net_debt": 100.0,
                "shares_outstanding": 50.0,
                "multiple_type": "ev_ebitda",
            }
        )
        assert result == pytest.approx(46.0, abs=1e-4)

    def test_ev_ebitda_with_net_cash(self):
        # (200 * 12 - (-50)) / 50 = (2400 + 50) / 50 = 49.0
        result = compute_multiples_value.invoke(
            {
                "ntm_metric": 200.0,
                "multiple": 12.0,
                "net_debt": -50.0,  # net cash
                "shares_outstanding": 50.0,
                "multiple_type": "ev_ebitda",
            }
        )
        assert result == pytest.approx(49.0, abs=1e-4)

    def test_pe_known_value(self):
        # EPS * P/E = 5.0 * 20 = 100.0
        result = compute_multiples_value.invoke(
            {
                "ntm_metric": 5.0,
                "multiple": 20.0,
                "net_debt": 0.0,
                "shares_outstanding": 100.0,
                "multiple_type": "pe",
            }
        )
        assert result == pytest.approx(100.0, abs=1e-4)

    def test_fcf_yield_known_value(self):
        # ntm_fcf=500, shares=100 → fcf/share=5.0; price = 5.0 / 0.05 = 100.0
        result = compute_multiples_value.invoke(
            {
                "ntm_metric": 500.0,
                "multiple": 5.0,  # 5% yield
                "net_debt": 0.0,
                "shares_outstanding": 100.0,
                "multiple_type": "fcf_yield",
            }
        )
        assert result == pytest.approx(100.0, abs=1e-4)

    def test_zero_ntm_metric_returns_none(self):
        result = compute_multiples_value.invoke(
            {
                "ntm_metric": 0.0,
                "multiple": 12.0,
                "net_debt": 0.0,
                "shares_outstanding": 50.0,
                "multiple_type": "ev_ebitda",
            }
        )
        assert result is None

    def test_negative_ntm_metric_returns_none(self):
        result = compute_multiples_value.invoke(
            {
                "ntm_metric": -10.0,
                "multiple": 12.0,
                "net_debt": 0.0,
                "shares_outstanding": 50.0,
                "multiple_type": "pe",
            }
        )
        assert result is None

    def test_zero_multiple_returns_none(self):
        result = compute_multiples_value.invoke(
            {
                "ntm_metric": 100.0,
                "multiple": 0.0,
                "net_debt": 0.0,
                "shares_outstanding": 50.0,
                "multiple_type": "ev_ebitda",
            }
        )
        assert result is None

    def test_ev_ebitda_zero_shares_returns_none(self):
        result = compute_multiples_value.invoke(
            {
                "ntm_metric": 100.0,
                "multiple": 12.0,
                "net_debt": 0.0,
                "shares_outstanding": 0.0,
                "multiple_type": "ev_ebitda",
            }
        )
        assert result is None


# ── compute_scenario_weighted_value ──────────────────────────────────────────


@pytest.mark.unit
class TestComputeScenarioWeightedValue:
    # bull=150, base=120, bear=80; probs=0.25/0.50/0.25; price=100
    # prob_weighted = 150*0.25 + 120*0.50 + 80*0.25 = 37.5 + 60 + 20 = 117.5
    # upside_base = (120-100)/100 * 100 = 20%
    # upside_bull = (150-100)/100 * 100 = 50%
    # downside_bear = (80-100)/100 * 100 = -20%
    # rr = |20 / -20| = 1.0
    _BASE = dict(
        bull_nav=150.0,
        base_nav=120.0,
        bear_nav=80.0,
        bull_prob=0.25,
        base_prob=0.50,
        bear_prob=0.25,
        current_price=100.0,
    )

    def test_probability_weighted_nav(self):
        result = compute_scenario_weighted_value.invoke(self._BASE)
        assert result["probability_weighted_nav"] == pytest.approx(117.5, abs=1e-4)

    def test_upside_base_pct(self):
        result = compute_scenario_weighted_value.invoke(self._BASE)
        assert result["upside_base_pct"] == pytest.approx(20.0, abs=1e-4)

    def test_upside_bull_pct(self):
        result = compute_scenario_weighted_value.invoke(self._BASE)
        assert result["upside_bull_pct"] == pytest.approx(50.0, abs=1e-4)

    def test_downside_bear_pct(self):
        result = compute_scenario_weighted_value.invoke(self._BASE)
        assert result["downside_bear_pct"] == pytest.approx(-20.0, abs=1e-4)

    def test_risk_reward_ratio(self):
        result = compute_scenario_weighted_value.invoke(self._BASE)
        assert result["risk_reward_ratio"] == pytest.approx(1.0, abs=1e-4)

    def test_zero_current_price_returns_none_pcts(self):
        result = compute_scenario_weighted_value.invoke(
            {**self._BASE, "current_price": 0.0}
        )
        assert result["upside_base_pct"] is None
        assert result["upside_bull_pct"] is None
        assert result["downside_bear_pct"] is None
        assert result["risk_reward_ratio"] is None
        # prob_weighted_nav still computed
        assert result["probability_weighted_nav"] is not None

    def test_negative_current_price_returns_none_pcts(self):
        result = compute_scenario_weighted_value.invoke(
            {**self._BASE, "current_price": -10.0}
        )
        assert result["upside_base_pct"] is None

    def test_bear_equals_current_price_rr_is_none(self):
        # downside_bear = 0 → risk_reward_ratio = None
        result = compute_scenario_weighted_value.invoke(
            {**self._BASE, "bear_nav": 100.0}
        )
        assert result["downside_bear_pct"] == pytest.approx(0.0, abs=1e-6)
        assert result["risk_reward_ratio"] is None

    def test_high_conviction_scenario(self):
        # High base prob, large upside — should show high rr
        result = compute_scenario_weighted_value.invoke(
            dict(
                bull_nav=200.0,
                base_nav=180.0,
                bear_nav=90.0,
                bull_prob=0.20,
                base_prob=0.60,
                bear_prob=0.20,
                current_price=100.0,
            )
        )
        assert result["upside_base_pct"] == pytest.approx(80.0, abs=1e-4)
        assert result["downside_bear_pct"] == pytest.approx(-10.0, abs=1e-4)
        assert result["risk_reward_ratio"] == pytest.approx(8.0, abs=1e-4)

    def test_output_schema_in_extras(self):
        schema = compute_scenario_weighted_value.extras["output_schema"]
        assert "properties" in schema
        for key in (
            "probability_weighted_nav",
            "upside_base_pct",
            "upside_bull_pct",
            "downside_bear_pct",
            "risk_reward_ratio",
        ):
            assert key in schema["properties"]
