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

from muffin_agent.middlewares.agent_capture.tree import merge_subagent_tree


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

    # ── Tool-execution capture (declaring the channel opts this graph in) ────
    tool_runs: NotRequired[Annotated[list[dict[str, Any]], operator.add]]
    """Stage-level tool-execution records captured by ``AgentCaptureMiddleware``
    (classification / criteria_definition / valuation_methodology / synthesis).
    Per-criterion records ride inside each ``criterion_evaluations`` entry as
    its own ``tool_runs`` list (attached by the worker's ``package`` node)."""

    subagent_tree: NotRequired[Annotated[dict[str, Any], merge_subagent_tree]]
    """Stage-level sub-agent execution tree nodes captured by
    ``AgentCaptureMiddleware``, same scope as ``tool_runs`` above. Per-criterion
    nodes ride inside each ``criterion_evaluations`` entry's own
    ``subagent_tree`` dict (attached by the worker's ``package`` node)."""


class CriterionEvaluationSendPayload(TypedDict):
    """Payload shape for one ``Send`` to the ``criterion_evaluation`` worker.

    Carries everything needed to evaluate a single criterion in
    isolation.  Doubles as the worker subgraph's input schema.
    """

    ticker: str
    query: str
    criterion: dict[str, Any]
    """One ``ValuationCriterion`` dict (from ``merged_criteria``) plus a
    ``source`` tag (``"skill"`` or ``"web"``)."""

    classification: dict[str, Any]
    """Full classification payload for context."""


# ── Explicit node input schemas (compiled-agent-as-node pattern) ─────────────
#
# Each compiled stage agent is added via ``graph.add_node(name, agent,
# input_schema=<one of these>)``.  NEVER pass ``agent.input_schema`` — it is
# a property-less ``RootModel`` that maps ``{}`` and raises at coercion (see
# the compiled-subagent composition rules in CLAUDE.md).


class TickerClassificationInput(TypedDict, total=False):
    """Fields the Stage 1 classification agent reads from the outer state."""

    ticker: str
    query: str


class CriteriaDefinitionInput(TypedDict, total=False):
    """Fields the Stage 2 criteria definition agent reads from the outer state.

    The flat classification keys feed both the runtime prompt and
    ``SkillFilterMiddleware[TickerClassification]`` (which reads them off
    the agent's own state).
    """

    ticker: str
    query: str
    sector: str
    sub_sector: str
    market: str
    stock_type: str


class ValuationMethodologyInput(TypedDict, total=False):
    """Fields the Stage 3 methodology agent reads from the outer state."""

    ticker: str
    query: str
    sector: str
    sub_sector: str
    market: str
    stock_type: str
    classification: dict[str, Any]


class SynthesisInput(TypedDict, total=False):
    """Fields the Stage 5 synthesis agent reads from the outer state."""

    ticker: str
    query: str
    classification: dict[str, Any]
    criteria_definition: dict[str, Any]
    valuation_methodology: dict[str, Any]
    merged_criteria: list[dict[str, Any]]
    criterion_evaluations: list[dict[str, Any]]
