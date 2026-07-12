"""Shared schemas for the persona council.

Universal ``AnalystSignal`` base model that every persona / specialist node
output must conform to so the council judge can ensemble them, plus shared
sub-models reused across persona evidence dicts.

The 5-tier ``InvestmentSignal`` vocabulary is intentionally re-declared here
(rather than imported from ``trading_decision``) — personas are independent
of the trading_decision pipeline.  The literal values must stay in lockstep
with ``trading_decision.schemas.InvestmentSignal`` and
``criteria_analysis.schemas.SynthesisSignal`` so all three pipelines share
one rating language.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field
from typing_extensions import TypedDict

# Re-exported so persona files (which already import from here) can annotate their
# `tool_runs` state channel with the SAME reducer the collect_data agent's
# AgentCaptureState uses — a different reducer raises "Channel already exists with
# a different type" when create_agent merges the schemas.
from muffin_agent.middlewares.agent_capture.records import (
    merge_tool_runs as merge_tool_runs,
)

# ── Shared node input contract ────────────────────────────────────────────────


class PersonaInput(TypedDict, total=False):
    """Input contract for any council-eligible persona / specialist node.

    Every persona / specialist subgraph reads exactly these three fields, so
    the council passes them via ``add_node(slug, agent, input_schema=PersonaInput)``.

    **Why an explicit schema (not ``agent.input_schema``):** a ``create_agent``
    subagent's ``.input_schema`` is a property-less ``RootModel`` that does NOT
    reflect the state schema's ``OmitFromSchema(input=False)`` fields, so passing
    it makes LangGraph map ``{}`` into the node and raise at coercion. Handing a
    real field-based schema is what actually expresses the per-node input
    contract (and isolates the node to just these fields).
    """

    ticker: str
    as_of_date: str
    query: str | None


# ── Shared rating vocabulary ──────────────────────────────────────────────────

InvestmentSignal = Literal[
    "strong_sell",
    "sell",
    "hold",
    "buy",
    "strong_buy",
]
"""5-tier conviction scale. Mirror of ``trading_decision.InvestmentSignal``
and ``criteria_analysis.SynthesisSignal`` so the persona council, the
trading-decision pipeline, and the criteria-driven analysis all speak one
rating language."""


# ── Score detail sub-model (reused across persona evidence) ──────────────────


class ScoreDetail(BaseModel):
    """A single sub-score with its rationale.

    Every persona breaks its overall verdict into 3–8 sub-scores (e.g.
    fundamentals, moat, valuation).  This shape lets the persona evidence
    expose those sub-scores uniformly so the council judge can compare
    across personas (e.g. who scored moat highest?).
    """

    score: float
    """Raw score for this sub-dimension (scale defined by the persona)."""

    max_score: float
    """Maximum achievable score for this sub-dimension; useful when
    personas use different total scales (e.g. Munger uses 0–10 per
    dimension, Buffett uses 0–27 total)."""

    details: str
    """Human-readable rationale citing the specific metrics that
    contributed to the score."""

    metrics: dict[str, Any] = Field(default_factory=dict)
    """Optional raw metric values used in the score; lets the council judge
    drill into ``metrics["roe"]`` etc. without re-fetching data."""


# ── Universal AnalystSignal base ──────────────────────────────────────────────


class AnalystSignal(BaseModel):
    """Universal output contract for any council-eligible agent.

    Persona-specific schemas extend this and narrow ``evidence`` to a typed
    Pydantic model.  The council judge consumes ``list[AnalystSignal]`` and
    synthesises ``CouncilSynthesisOutput`` from the ensemble.
    """

    agent_id: str
    """Stable slug identifying which persona / specialist produced this
    signal (e.g. ``"warren_buffett"``, ``"technicals"``).  Used by the
    council judge to attribute votes and break down dissent."""

    signal: InvestmentSignal
    """5-tier rating.  Mapping from the persona's internal score to this
    rating is defined in each persona's prompt — typically high score +
    high conviction maps to ``strong_buy``, etc."""

    confidence: float = Field(ge=0.0, le=1.0)
    """Confidence in the rating, 0.0–1.0.  Independent of direction:
    a persona can be very confident in ``strong_sell``."""

    reasoning: str
    """1–3 sentence persona-voiced explanation citing the specific
    sub-scores or evidence that drove the rating."""

    evidence: dict[str, Any] = Field(default_factory=dict)
    """Persona-specific facts dict (sub-scores, computed values, flags).
    Narrowed to a typed Pydantic sub-model in persona-specific signals
    (e.g. ``WarrenBuffettSignal.evidence: WarrenBuffettEvidence``)."""
