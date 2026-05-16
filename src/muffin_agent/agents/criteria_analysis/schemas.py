"""Pydantic schemas used across the criteria-driven analysis orchestrator.

Each stage of the orchestrator emits a structured output that downstream
stages consume.  Per-stage shapes live here together so the graph node
files stay focused on agent assembly and the LangGraph wiring.

Reuses:
- ``ValuationCriterion`` from ``criteria_definition`` for individual
  criteria.
- ``CriterionEvaluationOutput`` / ``SubCriterion`` from
  ``criterion_evaluation`` for per-criterion scoring.
- ``DataSource`` from ``investment.schemas`` for source attribution.
"""

from typing import Literal

from pydantic import BaseModel, Field

from ..criteria_definition import ValuationCriterion
from ..investment.schemas import DataSource

# ── Stage 1: ticker classification ────────────────────────────────────────────


class TickerClassificationOutput(BaseModel):
    """Structured output of the Stage 1 ticker classification agent.

    Emits the four classification fields used by
    ``SkillFilterMiddleware[TickerClassification]`` to filter
    ``/skills/valuation/`` downstream.
    """

    ticker: str
    sector: str
    """Sector tag, e.g. ``'banking'`` or ``'software-saas'``."""

    sub_sector: str | None = None
    """Optional finer split (e.g. ``'life'`` for insurance)."""

    market: Literal["developed", "emerging"]
    stock_type: Literal["value", "growth"]

    rationale: str
    """2–4 sentences explaining the classification."""

    confidence: float
    """0.0–1.0 reflecting classification certainty."""

    data_sources: list[DataSource] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)


# ── Stage 3: valuation methodology research ───────────────────────────────────


class ValuationMethodologyOutput(BaseModel):
    """Structured output of the Stage 3 valuation methodology agent.

    Surfaces the canonical valuation approach for the ticker plus any
    criteria the analyst community considers material that the
    skill-filtered stream would not otherwise capture.
    """

    ticker: str

    methodology_summary: str
    """2–4 paragraphs naming the canonical approach (e.g. 'DCF + EV/EBITDA
    peer multiple') and why it fits this ticker."""

    additional_criteria: list[ValuationCriterion]
    """Criteria not commonly captured in skills (recent sell-side debate,
    current narrative drivers, idiosyncratic governance issues)."""

    sources: list[DataSource] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)


# ── Stage 5: synthesis ────────────────────────────────────────────────────────


SynthesisSignal = Literal[
    "strong_sell",
    "sell",
    "hold",
    "buy",
    "strong_buy",
]


class WeightedBreakdownEntry(BaseModel):
    """One row in the per-criterion contribution table."""

    name: str
    weight: float
    score: float
    contribution: float
    """``weight * score``."""

    source: Literal["skill", "web"]
    """Which upstream stage produced this criterion."""


class CriteriaAnalysisSynthesis(BaseModel):
    """Final synthesis output of the orchestrator."""

    ticker: str

    composite_score: float
    """Weighted average of criterion scores (0.0–1.0)."""

    signal: SynthesisSignal

    weighted_breakdown: list[WeightedBreakdownEntry]
    """Per-criterion weight × score table."""

    key_positives: list[str]
    """Top ~3 strongest supportive evidence points."""

    key_negatives: list[str]
    """Top ~3 strongest opposing evidence points."""

    divergences: list[str] = Field(default_factory=list)
    """Cases where sub-scores conflict materially (e.g. strong fundamentals
    but adverse macro regime)."""

    confidence: float
    """0.0–1.0; floor of weighted criterion confidences and classification
    confidence."""

    thesis_paragraph: str
    """1–2 paragraph investment thesis."""
