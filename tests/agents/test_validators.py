"""Tests for investment output semantic validators."""

import pytest

from muffin_agent.agents.investment.validators import (
    get_validator,
    validate_company_analysis_output,
    validate_forecast_output,
)

# ---------------------------------------------------------------------------
# get_validator dispatch
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestGetValidator:
    """Verify validator registry dispatch."""

    def test_forecast_output(self):
        class ForecastOutput:
            pass

        assert get_validator(ForecastOutput) is validate_forecast_output

    def test_company_analysis_output(self):
        class CompanyAnalysisOutput:
            pass

        assert get_validator(CompanyAnalysisOutput) is validate_company_analysis_output

    def test_unknown_schema_returns_none(self):
        class UnknownSchema:
            pass

        assert get_validator(UnknownSchema) is None


# ---------------------------------------------------------------------------
# ForecastOutput validation
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestForecastValidation:
    """Validate ForecastOutput semantic rules."""

    def _make_scenario(self, label, probability, years=None):
        projections = [{"year": y} for y in (years or [2027, 2028, 2029])]
        return {
            "label": label,
            "probability": probability,
            "key_assumptions": [],
            "probability_rationale": "test",
            "narrative": "test",
            "projections": projections,
        }

    def test_valid_forecast_no_warnings(self):
        data = {
            "base_case": self._make_scenario("base", 0.60),
            "bull_case": self._make_scenario("bull", 0.25),
            "bear_case": self._make_scenario("bear", 0.15),
            "confidence": 0.7,
            "limitations": ["a", "b"],
        }
        assert validate_forecast_output(data) == []

    def test_probability_sum_too_high(self):
        data = {
            "base_case": self._make_scenario("base", 0.70),
            "bull_case": self._make_scenario("bull", 0.30),
            "bear_case": self._make_scenario("bear", 0.20),
        }
        warnings = validate_forecast_output(data)
        assert any("sum to 1.20" in w for w in warnings)

    def test_probability_sum_too_low(self):
        data = {
            "base_case": self._make_scenario("base", 0.40),
            "bull_case": self._make_scenario("bull", 0.20),
            "bear_case": self._make_scenario("bear", 0.10),
        }
        warnings = validate_forecast_output(data)
        assert any("sum to 0.70" in w for w in warnings)

    def test_probability_sum_within_tolerance(self):
        data = {
            "base_case": self._make_scenario("base", 0.58),
            "bull_case": self._make_scenario("bull", 0.25),
            "bear_case": self._make_scenario("bear", 0.15),
        }
        # 0.98, within ±0.05
        warnings = validate_forecast_output(data)
        assert not any("probabilities sum" in w for w in warnings)

    def test_projections_unsorted(self):
        data = {
            "base_case": self._make_scenario("base", 0.60, years=[2029, 2027, 2028]),
            "bull_case": self._make_scenario("bull", 0.25),
            "bear_case": self._make_scenario("bear", 0.15),
        }
        warnings = validate_forecast_output(data)
        assert any("base_case projections are not sorted" in w for w in warnings)

    def test_high_confidence_many_limitations(self):
        data = {
            "base_case": self._make_scenario("base", 0.60),
            "bull_case": self._make_scenario("bull", 0.25),
            "bear_case": self._make_scenario("bear", 0.15),
            "confidence": 0.95,
            "limitations": ["a", "b", "c"],
        }
        warnings = validate_forecast_output(data)
        assert any("confidence=0.95 seems high" in w for w in warnings)


# ---------------------------------------------------------------------------
# CompanyAnalysisOutput validation
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCompanyAnalysisValidation:
    """Validate CompanyAnalysisOutput semantic rules."""

    def test_valid_company_no_warnings(self):
        data = {
            "financial_history": {"years": [2021, 2022, 2023, 2024]},
            "company_signal": "pass",
            "financial_quality": {"quality_signal": "high"},
            "confidence": 0.7,
            "limitations": ["a"],
        }
        assert validate_company_analysis_output(data) == []

    def test_years_unsorted(self):
        data = {
            "financial_history": {"years": [2024, 2022, 2023]},
        }
        warnings = validate_company_analysis_output(data)
        assert any("not sorted" in w for w in warnings)

    def test_years_duplicates(self):
        data = {
            "financial_history": {"years": [2022, 2023, 2023, 2024]},
        }
        warnings = validate_company_analysis_output(data)
        assert any("duplicates" in w for w in warnings)

    def test_signal_consistency_pass_with_distressed(self):
        data = {
            "company_signal": "pass",
            "financial_quality": {"quality_signal": "distressed"},
        }
        warnings = validate_company_analysis_output(data)
        assert any("unusual" in w for w in warnings)

    def test_signal_consistency_fail_with_high(self):
        data = {
            "company_signal": "fail",
            "financial_quality": {"quality_signal": "high"},
        }
        warnings = validate_company_analysis_output(data)
        assert any("unusual" in w for w in warnings)

    def test_signal_consistency_watch_with_adequate(self):
        data = {
            "company_signal": "watch",
            "financial_quality": {"quality_signal": "adequate"},
        }
        warnings = validate_company_analysis_output(data)
        assert not any("unusual" in w for w in warnings)

    def test_high_confidence_many_limitations(self):
        data = {
            "confidence": 0.90,
            "limitations": ["a", "b", "c", "d"],
        }
        warnings = validate_company_analysis_output(data)
        assert any("confidence=0.90 seems high" in w for w in warnings)

    def test_low_confidence_many_limitations_no_warning(self):
        data = {
            "confidence": 0.60,
            "limitations": ["a", "b", "c", "d"],
        }
        warnings = validate_company_analysis_output(data)
        assert not any("confidence" in w for w in warnings)
