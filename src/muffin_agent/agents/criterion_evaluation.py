"""Criterion evaluation agent.

Deep agent that evaluates a single investment criterion by orchestrating
data collection, validation, and scoring with reflection.

Structured output is enforced via :class:`CriterionEvaluationOutput` so
upstream orchestrators can consume the result programmatically without
parsing a free-text block.
"""

from typing import Literal

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


# ── Agent factory ─────────────────────────────────────────────────────────────


async def create_criterion_evaluation_agent(config: RunnableConfig):
    """Build the criterion evaluation deep agent.

    Create a deep agent that evaluates a single investment criterion by
    collecting targeted data, validating it, scoring the criterion, and
    reflecting on the evaluation quality.

    ``response_format=AutoStrategy(CriterionEvaluationOutput)`` instructs
    the agent to call a structured output tool as its final act so callers
    receive a validated Pydantic model in
    ``result["structured_response"]``.
    """
    subagents = await build_analysis_subagents(config)
    configuration = ModelConfiguration.from_runnable_config(config)
    primary, *fallbacks = configuration.get_llm_for_role("orchestrator")
    summariser = configuration.get_summariser()

    builder = (
        MuffinAgentBuilder(primary, name="criterion_evaluation")
        .with_system_prompt_template("criterion_evaluation.jinja")
        .with_fallback_models(*fallbacks)
        .with_sandbox()
        .with_short_term_memory()
        .with_persistent_memory()
        .with_subagents(subagents)
        .with_response_format(AutoStrategy(schema=CriterionEvaluationOutput))
    )
    if summariser is not None:
        builder = builder.with_tool_knowledge(summariser)
    return builder.build_deep_agent()
