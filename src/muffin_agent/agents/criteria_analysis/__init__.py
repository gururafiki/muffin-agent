"""Criteria-driven investment analysis orchestrator.

Top-level pipeline that classifies a ticker, runs criteria_definition
(skill-filtered) and valuation_methodology (web-research) in parallel,
fans each merged criterion out to ``criterion_evaluation``, and
synthesises a final investment view.
"""

from .graph import build_criteria_analysis_graph
from .schemas import (
    CriteriaAnalysisSynthesis,
    TickerClassificationOutput,
    ValuationMethodologyOutput,
    WeightedBreakdownEntry,
)

__all__ = [
    "CriteriaAnalysisSynthesis",
    "TickerClassificationOutput",
    "ValuationMethodologyOutput",
    "WeightedBreakdownEntry",
    "build_criteria_analysis_graph",
]
