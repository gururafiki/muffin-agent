"""Unit tests for macro tools (yield curve, factor Z-scores, VIX regime)."""

import json

import pytest

from muffin_agent.tools.macro import (
    compute_factor_zscore,
    compute_vix_regime,
    compute_yield_curve_metrics,
)


@pytest.mark.unit
class TestVIXRegimeTool:
    def test_complacency(self):
        assert compute_vix_regime.invoke({"vix_level": 12.0}) == "complacency"

    def test_normal(self):
        assert compute_vix_regime.invoke({"vix_level": 17.0}) == "normal"

    def test_elevated(self):
        assert compute_vix_regime.invoke({"vix_level": 25.0}) == "elevated"

    def test_crisis(self):
        assert compute_vix_regime.invoke({"vix_level": 35.0}) == "crisis"


@pytest.mark.unit
class TestFactorZscoreTool:
    def test_basic(self):
        result = compute_factor_zscore.invoke(
            {
                "factor_name": "HML",
                "trailing_12m": 0.10,
                "mean_60m": 0.05,
                "std_60m": 0.02,
            }
        )
        parsed = json.loads(result)
        assert parsed["factor_name"] == "HML"
        assert parsed["z_score"] == pytest.approx(2.5)

    def test_zero_std(self):
        result = compute_factor_zscore.invoke(
            {
                "factor_name": "SMB",
                "trailing_12m": 0.10,
                "mean_60m": 0.05,
                "std_60m": 0.0,
            }
        )
        parsed = json.loads(result)
        assert parsed["z_score"] is None


@pytest.mark.unit
class TestYieldCurveMetricsTool:
    def test_full(self):
        result = compute_yield_curve_metrics.invoke(
            {
                "yield_10y": 4.25,
                "yield_2y": 4.0,
                "yield_3m": 5.0,
                "tips_breakeven_10y": 2.3,
                "effr": 5.25,
            }
        )
        parsed = json.loads(result)
        assert parsed["slope_10y2y_bps"] == pytest.approx(25.0)
        assert parsed["slope_10y3m_bps"] == pytest.approx(-75.0)
        assert parsed["real_yield_10y_bps"] == pytest.approx(195.0)
        assert parsed["policy_rate_distance_bps"] == pytest.approx(275.0)

    def test_partial(self):
        result = compute_yield_curve_metrics.invoke(
            {"yield_10y": 4.25, "yield_2y": 4.0}
        )
        parsed = json.loads(result)
        assert parsed["slope_10y2y_bps"] == pytest.approx(25.0)
        assert parsed["slope_10y3m_bps"] is None
