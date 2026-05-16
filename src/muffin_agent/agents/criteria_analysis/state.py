"""State schemas for the criteria-driven analysis orchestrator.

- ``CriteriaAnalysisState`` — outer state flowing through the orchestrator
  graph.  Each stage writes its own key; ``criterion_evaluations`` uses
  ``operator.add`` so per-criterion fan-out workers can each append.
- ``CriterionEvaluationSendPayload`` — payload shape carried by each
  ``Send`` from ``merge_criteria`` to ``criterion_evaluation``.

The flat ``sector`` / ``sub_sector`` / ``market`` / ``stock_type`` keys
are intentional: ``SkillFilterMiddleware[TickerClassification]`` (used
inside ``criteria_definition``) reads them off the top level of the
state dict.  Stage 1 writes them; the CLI can also pre-supply them.
"""

import operator
from typing import Annotated, Any, NotRequired

from typing_extensions import TypedDict


class CriteriaAnalysisState(TypedDict):
    """Outer state of the criteria-driven analysis orchestrator graph."""

    # ── Input ────────────────────────────────────────────────────────────────
    ticker: str
    query: str

    # ── Pre-classification overrides (also written by Stage 1) ───────────────
    sector: NotRequired[str]
    sub_sector: NotRequired[str]
    market: NotRequired[str]
    stock_type: NotRequired[str]

    # ── Stage 1 ──────────────────────────────────────────────────────────────
    classification: dict[str, Any]
    """Full ``TickerClassificationOutput.model_dump()`` for downstream context."""

    # ── Stage 2 (parallel with Stage 3) ──────────────────────────────────────
    criteria_definition: dict[str, Any]
    """``CriteriaDefinitionOutput.model_dump()``."""

    # ── Stage 3 (parallel with Stage 2) ──────────────────────────────────────
    valuation_methodology: dict[str, Any]
    """``ValuationMethodologyOutput.model_dump()``."""

    # ── Stage 4a ─────────────────────────────────────────────────────────────
    merged_criteria: list[dict[str, Any]]
    """Deduplicated criteria with ``source`` tag.  One Send is emitted per
    entry to the criterion_evaluation fan-out."""

    # ── Stage 4b fan-in accumulator ──────────────────────────────────────────
    criterion_evaluations: Annotated[list[dict[str, Any]], operator.add]
    """Per-criterion ``CriterionEvaluationOutput.model_dump()`` results,
    one appended by each parallel ``criterion_evaluation`` worker."""

    # ── Stage 5 ──────────────────────────────────────────────────────────────
    synthesis: dict[str, Any]
    """``CriteriaAnalysisSynthesis.model_dump()``."""


class CriterionEvaluationSendPayload(TypedDict):
    """Payload shape for one ``Send`` to the ``criterion_evaluation`` node.

    Carries everything needed to evaluate a single criterion in
    isolation.  ``criterion_evaluations`` is included as an empty list so
    the LangGraph reducer recognises the field on the per-Send substate.
    """

    ticker: str
    query: str
    criterion: dict[str, Any]
    """One ``ValuationCriterion`` dict (from ``merged_criteria``) plus a
    ``source`` tag (``"skill"`` or ``"web"``)."""

    classification: dict[str, Any]
    """Full classification payload for context."""

    criterion_evaluations: list[dict[str, Any]]
