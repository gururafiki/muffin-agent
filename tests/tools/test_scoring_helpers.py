"""Unit tests for persona scoring helpers."""

import math

import pytest

from muffin_agent.tools.scoring_helpers import (
    Score,
    compute_buffett_3stage_dcf,
    compute_damodaran_fcff_dcf,
    compute_graham_number,
    compute_intrinsic_value_dcf,
    compute_ncav_per_share,
    compute_owner_earnings,
    compute_peg_ratio,
    compute_price_momentum,
    compute_volatility_metrics,
    score_current_ratio,
    score_debt_to_equity,
    score_eps_cagr,
    score_fcf_yield,
    score_insider_buy_ratio,
    score_margin_stability,
    score_operating_margin,
    score_revenue_cagr,
    score_roe,
)


@pytest.mark.unit
class TestScoreROE:
    def test_excellent_roe(self):
        assert score_roe(0.25).score == 3

    def test_strong_roe(self):
        assert score_roe(0.18).score == 2

    def test_decent_roe(self):
        assert score_roe(0.12).score == 1

    def test_weak_roe(self):
        assert score_roe(0.05).score == 0

    def test_missing_roe(self):
        result = score_roe(None)
        assert result.score == 0
        assert "not available" in result.details


@pytest.mark.unit
class TestScoreDebtToEquity:
    @pytest.mark.parametrize(
        ("de", "expected_score"),
        [(0.1, 3), (0.5, 2), (1.0, 1), (2.0, 0), (None, 0)],
    )
    def test_thresholds(self, de, expected_score):
        assert score_debt_to_equity(de).score == expected_score


@pytest.mark.unit
class TestScoreOperatingMargin:
    @pytest.mark.parametrize(
        ("om", "expected"),
        [(0.25, 2), (0.18, 1), (0.10, 0), (None, 0)],
    )
    def test_thresholds(self, om, expected):
        assert score_operating_margin(om).score == expected


@pytest.mark.unit
class TestScoreCurrentRatio:
    @pytest.mark.parametrize(
        ("cr", "expected"), [(2.5, 2), (1.7, 1), (1.0, 0), (None, 0)]
    )
    def test_thresholds(self, cr, expected):
        assert score_current_ratio(cr).score == expected


@pytest.mark.unit
class TestScoreFCFYield:
    def test_high_yield(self):
        # FCF 15 on market cap 100 = 15%
        assert score_fcf_yield(15.0, 100.0).score == 4

    def test_strong_yield(self):
        # 7%
        assert score_fcf_yield(7.0, 100.0).score == 3

    def test_negative_yield(self):
        assert score_fcf_yield(-5.0, 100.0).score == 0

    def test_missing_inputs(self):
        assert score_fcf_yield(None, 100.0).score == 0
        assert score_fcf_yield(15.0, None).score == 0
        assert score_fcf_yield(15.0, 0).score == 0


@pytest.mark.unit
class TestScoreRevenueCAGR:
    def test_strong_growth(self):
        # 100 → 250 over 5 periods (4y CAGR ~ 25%)
        result = score_revenue_cagr([100, 130, 165, 205, 250])
        assert result.score == 3

    def test_moderate_growth(self):
        # 100 → 160 over 5 periods (~12% CAGR)
        result = score_revenue_cagr([100, 112, 125, 140, 160])
        assert result.score == 2

    def test_insufficient_data(self):
        assert score_revenue_cagr([100]).score == 0
        assert score_revenue_cagr([]).score == 0

    def test_zero_oldest(self):
        assert score_revenue_cagr([0, 100, 200]).score == 0

    def test_drops_none(self):
        # Same as the 25% CAGR test above but with None entries scattered through
        result = score_revenue_cagr([None, 100, 130, None, 165, 205, 250])
        assert result.score == 3


@pytest.mark.unit
class TestScoreEPSCAGR:
    def test_strong_growth(self):
        result = score_eps_cagr([1.0, 1.3, 1.7, 2.2, 2.5])
        assert result.score == 3

    def test_negative_latest_eps(self):
        result = score_eps_cagr([1.0, 1.5, -0.5])
        assert result.score == 0
        assert "negative" in result.details

    def test_negative_oldest_eps(self):
        result = score_eps_cagr([-1.0, 1.5])
        assert result.score == 0
        assert "Insufficient" in result.details


@pytest.mark.unit
class TestScoreInsiderBuyRatio:
    def test_heavy_buying(self):
        trades = [{"transaction_shares": 100} for _ in range(8)] + [
            {"transaction_shares": -50} for _ in range(2)
        ]
        result = score_insider_buy_ratio(trades)
        assert result.score == 8

    def test_balanced_activity(self):
        trades = [{"transaction_shares": 100} for _ in range(5)] + [
            {"transaction_shares": -50} for _ in range(5)
        ]
        result = score_insider_buy_ratio(trades)
        assert result.score == 6

    def test_net_selling(self):
        trades = [{"transaction_shares": 100} for _ in range(2)] + [
            {"transaction_shares": -50} for _ in range(8)
        ]
        result = score_insider_buy_ratio(trades)
        assert result.score == 4

    def test_empty_list(self):
        # ai-hedge-fund defaults to neutral 5 when no insider data
        assert score_insider_buy_ratio([]).score == 5


@pytest.mark.unit
class TestScoreMarginStability:
    def test_stable_margins(self):
        margins = [0.20, 0.21, 0.205, 0.198, 0.207]
        assert score_margin_stability(margins).score == 2

    def test_unstable_margins(self):
        margins = [0.20, 0.05, 0.30, 0.10, 0.25]
        assert score_margin_stability(margins).score == 0

    def test_insufficient_data(self):
        assert score_margin_stability([0.2, 0.21]).score == 0


@pytest.mark.unit
class TestComputeOwnerEarnings:
    def test_basic(self):
        # NI 100, D&A 20, capex 30 -> 100 + 20 - 30*0.75 = 97.5
        assert compute_owner_earnings(100, 20, 30) == 97.5

    def test_custom_maintenance_ratio(self):
        # 100 + 20 - 30*1.0 = 90
        assert compute_owner_earnings(100, 20, 30, maintenance_capex_ratio=1.0) == 90

    def test_missing_inputs(self):
        assert compute_owner_earnings(None, 20, 30) is None
        assert compute_owner_earnings(100, None, 30) is None
        assert compute_owner_earnings(100, 20, 0) is None


@pytest.mark.unit
class TestComputeGrahamNumber:
    def test_basic(self):
        # sqrt(22.5 * 5 * 20) = sqrt(2250) ≈ 47.43
        assert compute_graham_number(5.0, 20.0) == pytest.approx(47.434, abs=0.01)

    def test_negative_eps(self):
        assert compute_graham_number(-1.0, 20.0) is None

    def test_negative_bvps(self):
        assert compute_graham_number(5.0, -1.0) is None


@pytest.mark.unit
class TestComputeNCAVPerShare:
    def test_basic(self):
        # (1000 - 400) / 100 = 6.0
        assert compute_ncav_per_share(1000, 400, 100) == 6.0

    def test_negative_ncav(self):
        # (300 - 800) / 100 = -5.0 (more liabilities than current assets)
        assert compute_ncav_per_share(300, 800, 100) == -5.0

    def test_missing_inputs(self):
        assert compute_ncav_per_share(None, 400, 100) is None
        assert compute_ncav_per_share(1000, 400, 0) is None


@pytest.mark.unit
class TestComputePEGRatio:
    def test_attractive_peg(self):
        # P/E 15, growth 20% → PEG = 15 / 20 = 0.75
        assert compute_peg_ratio(15.0, 0.20) == 0.75

    def test_expensive_peg(self):
        # P/E 30, growth 10% → PEG = 30 / 10 = 3.0
        assert compute_peg_ratio(30.0, 0.10) == 3.0

    def test_negative_growth(self):
        assert compute_peg_ratio(15.0, -0.05) is None

    def test_negative_pe(self):
        assert compute_peg_ratio(-10.0, 0.10) is None


@pytest.mark.unit
class TestComputeIntrinsicValueDCF:
    def test_terminal_gt_discount_returns_none(self):
        assert compute_intrinsic_value_dcf(100, 0.10, 0.05, 0.10) is None

    def test_zero_base_cf_returns_none(self):
        assert compute_intrinsic_value_dcf(0, 0.10, 0.10, 0.02) is None

    def test_basic_dcf_positive(self):
        # base 100, growth 10%, disc 10%, terminal 2.5% — produces positive value
        val = compute_intrinsic_value_dcf(100, 0.10, 0.10, 0.025, years=5)
        assert val is not None
        assert val > 100  # intrinsic value > current CF


@pytest.mark.unit
class TestComputeBuffett3StageDCF:
    def test_returns_positive_value(self):
        val = compute_buffett_3stage_dcf(100)
        assert val is not None
        assert val > 100

    def test_haircut_applied(self):
        no_haircut = compute_buffett_3stage_dcf(100, conservatism_factor=1.0)
        with_haircut = compute_buffett_3stage_dcf(100, conservatism_factor=0.85)
        assert no_haircut is not None and with_haircut is not None
        assert with_haircut == pytest.approx(no_haircut * 0.85, rel=1e-9)


@pytest.mark.unit
class TestComputeDamodaranFCFFDCF:
    def test_returns_tuple(self):
        result = compute_damodaran_fcff_dcf(
            base_fcff=100.0, initial_growth=0.08, beta=1.2
        )
        assert result is not None
        value, discount_rate = result
        # discount = 4% + 1.2 × 5% = 10%
        assert discount_rate == pytest.approx(0.10, abs=1e-9)
        assert value > 100

    def test_growth_capped(self):
        # initial_growth 25% should be capped at 12% internally
        result_25 = compute_damodaran_fcff_dcf(100.0, 0.25, 1.0)
        result_12 = compute_damodaran_fcff_dcf(100.0, 0.12, 1.0)
        assert result_25 is not None and result_12 is not None
        assert result_25[0] == pytest.approx(result_12[0], rel=1e-9)


@pytest.mark.unit
class TestComputeVolatilityMetrics:
    def test_returns_all_fields(self):
        returns = [0.01, -0.02, 0.005, 0.015, -0.01, 0.005, -0.008] * 10
        result = compute_volatility_metrics(returns)
        assert result["annualized_volatility"] is not None
        assert result["skewness"] is not None
        assert result["excess_kurtosis"] is not None
        assert result["max_drawdown_pct"] is not None

    def test_short_series(self):
        result = compute_volatility_metrics([0.01, 0.02])
        assert result["annualized_volatility"] is None
        assert result["skewness"] is None

    def test_drawdown_calculation(self):
        # Mostly negative returns → significant drawdown
        returns = [-0.05] * 20
        result = compute_volatility_metrics(returns)
        assert result["max_drawdown_pct"] is not None
        assert result["max_drawdown_pct"] < 0


@pytest.mark.unit
class TestComputePriceMomentum:
    def test_uptrend(self):
        prices = [100 + i for i in range(20)]
        result = compute_price_momentum(prices)
        assert result["total_return_pct"] is not None
        assert result["total_return_pct"] > 0

    def test_downtrend(self):
        prices = [100 - i for i in range(20)]
        result = compute_price_momentum(prices)
        assert result["total_return_pct"] is not None
        assert result["total_return_pct"] < 0

    def test_short_series(self):
        result = compute_price_momentum([100, 105])
        assert result["total_return_pct"] is None


@pytest.mark.unit
class TestScoreDataclass:
    def test_score_is_frozen(self):
        s = Score(score=3, max_score=5, details="hi")
        with pytest.raises(Exception):  # FrozenInstanceError or AttributeError
            s.score = 4  # type: ignore[misc]

    def test_score_immutable_after_construction(self):
        s = Score(score=3, max_score=5, details="hi")
        # Use the score normally
        assert s.score + 1 == 4
        assert math.isclose(s.score, 3)
