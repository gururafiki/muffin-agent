"""Criterion evaluation agent.

Deep agent that evaluates a single investment criterion by orchestrating
data collection, validation, and scoring with reflection.

Structured output is enforced via :class:`CriterionEvaluationOutput` so
upstream orchestrators can consume the result programmatically without
parsing a free-text block.
"""

from typing import Annotated, Any, Literal

from deepagents import DeepAgentState
from langchain.agents.middleware.types import OmitFromSchema
from langchain.agents.structured_output import AutoStrategy
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from ..model_config import ModelConfiguration
from ..utils.agent_builder import MuffinAgentBuilder
from .investment.schemas import DataSource
from .subagents import build_analysis_subagents

# ── Output schema ─────────────────────────────────────────────────────────────


class SubCriterion(BaseModel):
    """One sub-criterion that contributes to an overall criterion score."""

    name: str
    weight: float
    """0.0–1.0; weights across sub-criteria should sum to ~1.0."""

    score: float
    """0.0–1.0."""

    evidence: str
    """1–2 sentences with specific data points."""


CriterionSignal = Literal[
    "strong_negative",
    "negative",
    "neutral",
    "positive",
    "strong_positive",
]


class CriterionEvaluationOutput(BaseModel):
    """Structured output of the criterion evaluation deep agent."""

    criterion_name: str
    score: float
    """0.0–1.0 overall criterion score."""

    confidence: float
    """0.0–1.0 reflecting data quality, NOT subjective certainty."""

    signal: CriterionSignal

    sub_criteria: list[SubCriterion]
    evidence_summary: list[str]
    """Bullet-style data points with source subagent + period."""

    reasoning: str
    """2–4 paragraphs explaining how data maps to the score."""

    counterargument: str
    """1–2 sentences describing the strongest argument against the score."""

    limitations: list[str] = Field(default_factory=list)
    data_sources: list[DataSource] = Field(default_factory=list)


class CriterionEvaluationNodeOutput(BaseModel):
    """Wrapper output for the criterion evaluation agent as a graph node.

    Single-field wrapper so ``_StructuredResponseToStateMiddleware``
    unpacks the structured response into the worker-local ``evaluation``
    channel; the worker's ``package`` node augments it with ``weight`` /
    ``source`` and appends it to the parent ``criterion_evaluations``
    accumulator.
    """

    evaluation: CriterionEvaluationOutput


# ── Agent state schema ────────────────────────────────────────────────────────


class CriterionEvaluationAgentState(DeepAgentState):
    """State schema for the criterion evaluation agent as a graph node.

    ``criterion`` and ``classification`` arrive as dicts from the Send
    payload; ``evaluation`` is written by the structured-response unpacker
    and flows OUT to the worker subgraph's ``package`` node.
    """

    ticker: Annotated[str, OmitFromSchema(input=False, output=True)]
    query: Annotated[str, OmitFromSchema(input=False, output=True)]
    criterion: Annotated[dict[str, Any], OmitFromSchema(input=False, output=True)]
    classification: Annotated[dict[str, Any], OmitFromSchema(input=False, output=True)]
    evaluation: Annotated[dict[str, Any], OmitFromSchema(input=True, output=False)]


# ── Agent factory ─────────────────────────────────────────────────────────────


async def create_criterion_evaluation_agent(config: RunnableConfig):
    """Build the criterion evaluation deep agent (compiled graph node).

    Added directly to the per-criterion worker subgraph as a node with
    ``input_schema=CriterionEvaluationSendPayload``. The task context
    (criterion, classification, ticker, query) is rendered into the system
    prompt per model call; the structured response unpacks into the
    ``evaluation`` channel.
    """
    subagents = await build_analysis_subagents(config)
    configuration = ModelConfiguration.from_runnable_config(config)
    primary, *fallbacks = configuration.get_llm_for_role("orchestrator")
    summariser = configuration.get_summariser()

    builder = (
        MuffinAgentBuilder(primary, name="criterion_evaluation")
        .with_state_schema(CriterionEvaluationAgentState)
        .with_runtime_system_prompt_template("criterion_evaluation.jinja")
        .with_fallback_models(*fallbacks)
        .with_sandbox()
        .with_short_term_memory()
        .with_persistent_memory()
        .with_subagents(subagents)
        .with_response_format(AutoStrategy(schema=CriterionEvaluationNodeOutput))
    )
    if summariser is not None:
        builder = builder.with_tool_knowledge(summariser)
    return builder.build_deep_agent()
