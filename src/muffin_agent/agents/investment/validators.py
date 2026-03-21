"""Post-processing semantic validators for investment output schemas.

Validates semantics that Pydantic type constraints cannot enforce (e.g.
probability sums, sorted arrays, cross-field consistency).  Returns a list
of warning strings — never modifies the data or raises.

Kept separate from the Pydantic schemas so the LLM-facing JSON schema stays
clean (no ``@model_validator`` side-effects that confuse ``AutoStrategy``).
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Registry: maps output-schema class name → validator function
# ---------------------------------------------------------------------------

_VALIDATORS: dict[str, type] = {}


def _register(cls_name: str):
    """Register a validator function for *cls_name*."""

    def decorator(fn):
        _VALIDATORS[cls_name] = fn
        return fn

    return decorator


def get_validator(schema_cls: type):
    """Return the validator function for *schema_cls*, or ``None``."""
    return _VALIDATORS.get(schema_cls.__name__)


# ---------------------------------------------------------------------------
# ForecastOutput
# ---------------------------------------------------------------------------

_PROB_TOLERANCE = 0.05


@_register("ForecastOutput")
def validate_forecast_output(data: dict[str, Any]) -> list[str]:
    """Validate ForecastOutput semantics."""
    warnings: list[str] = []

    # --- Scenario probability sum ---
    probs = []
    for key in ("base_case", "bull_case", "bear_case"):
        scenario = data.get(key)
        if isinstance(scenario, dict):
            p = scenario.get("probability")
            if isinstance(p, (int, float)):
                probs.append(p)
    if probs:
        total = sum(probs)
        if abs(total - 1.0) > _PROB_TOLERANCE:
            warnings.append(
                f"Scenario probabilities sum to {total:.2f}, expected ~1.0"
            )

    # --- Projections sorted by year ---
    for key in ("base_case", "bull_case", "bear_case"):
        scenario = data.get(key)
        if not isinstance(scenario, dict):
            continue
        projections = scenario.get("projections")
        if not isinstance(projections, list) or len(projections) < 2:
            continue
        years = [
            p["year"] for p in projections if isinstance(p, dict) and "year" in p
        ]
        if years != sorted(years):
            warnings.append(
                f"{key} projections are not sorted by year ascending"
            )

    # --- Confidence vs limitations ---
    warnings.extend(_check_confidence_vs_limitations(data))

    return warnings


# ---------------------------------------------------------------------------
# CompanyAnalysisOutput
# ---------------------------------------------------------------------------

_SIGNAL_QUALITY_MAP = {
    "pass": {"high", "adequate"},
    "watch": {"adequate", "low"},
    "fail": {"low", "distressed"},
}


@_register("CompanyAnalysisOutput")
def validate_company_analysis_output(data: dict[str, Any]) -> list[str]:
    """Validate CompanyAnalysisOutput semantics."""
    warnings: list[str] = []

    # --- Financial history years sorted, no duplicates ---
    fh = data.get("financial_history")
    if isinstance(fh, dict):
        years = fh.get("years")
        if isinstance(years, list) and len(years) >= 2:
            if years != sorted(years):
                warnings.append(
                    "financial_history.years are not sorted ascending"
                )
            if len(years) != len(set(years)):
                warnings.append(
                    "financial_history.years contain duplicates"
                )

    # --- company_signal consistency with financial_quality.quality_signal ---
    company_signal = data.get("company_signal")
    fq = data.get("financial_quality")
    if isinstance(fq, dict) and company_signal:
        quality_signal = fq.get("quality_signal")
        expected = _SIGNAL_QUALITY_MAP.get(company_signal)
        if expected and quality_signal and quality_signal not in expected:
            warnings.append(
                f"company_signal={company_signal!r} is unusual with "
                f"financial_quality.quality_signal={quality_signal!r}"
            )

    # --- Confidence vs limitations ---
    warnings.extend(_check_confidence_vs_limitations(data))

    return warnings


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_LIMITATIONS_THRESHOLD = 3
_CONFIDENCE_CAP = 0.8


def _check_confidence_vs_limitations(data: dict[str, Any]) -> list[str]:
    """Warn if high confidence despite many limitations."""
    limitations = data.get("limitations")
    confidence = data.get("confidence")
    if (
        isinstance(limitations, list)
        and len(limitations) >= _LIMITATIONS_THRESHOLD
        and isinstance(confidence, (int, float))
        and confidence > _CONFIDENCE_CAP
    ):
        return [
            f"confidence={confidence:.2f} seems high given "
            f"{len(limitations)} limitations listed"
        ]
    return []
