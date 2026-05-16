"""Stage 5: synthesis.

Reasoning-only deep agent (no subagents, no tools, no sandbox).
Reads classification, criteria_definition, valuation_methodology, and
the per-criterion evaluations and produces a final
``CriteriaAnalysisSynthesis`` with composite score, signal, weighted
breakdown, and thesis paragraph.
"""

from typing import Any

from langchain.agents.structured_output import AutoStrategy
from langchain_core.runnables import RunnableConfig
from langgraph.store.base import BaseStore
from typing_extensions import TypedDict

from ...model_config import ModelConfiguration
from ...utils.agent_builder import MuffinAgentBuilder
from ._node_helpers import invoke_structured_agent
from .schemas import CriteriaAnalysisSynthesis

# ── Input state schema ────────────────────────────────────────────────────────


class SynthesisInputState(TypedDict, total=False):
    """Fields read by ``synthesis_node`` from the outer state."""

    ticker: str
    query: str
    classification: dict[str, Any]
    criteria_definition: dict[str, Any]
    valuation_methodology: dict[str, Any]
    merged_criteria: list[dict[str, Any]]
    criterion_evaluations: list[dict[str, Any]]


# ── Agent factory ─────────────────────────────────────────────────────────────


async def create_synthesis_agent(
    config: RunnableConfig,
    store: BaseStore | None = None,
):
    """Build the synthesis deep agent — reasoning only, no subagents/tools."""
    configuration = ModelConfiguration.from_runnable_config(config)
    primary, *fallbacks = configuration.get_llm_for_role("reasoner")

    builder = (
        MuffinAgentBuilder(primary, name="criteria_analysis_synthesis")
        .with_system_prompt_template("criteria_analysis/synthesis.jinja")
        .with_fallback_models(*fallbacks)
        .with_short_term_memory()
        .with_response_format(AutoStrategy(schema=CriteriaAnalysisSynthesis))
    )
    if store is not None:
        builder = builder.with_store(store)
    return builder.build_deep_agent()


# ── Node ──────────────────────────────────────────────────────────────────────


async def synthesis_node(
    state: SynthesisInputState,
    config: RunnableConfig,
    *,
    store: BaseStore | None = None,
) -> dict[str, Any]:
    """Stage 5: synthesise the final investment view."""
    return await invoke_structured_agent(
        state=dict(state),
        config=config,
        agent_factory=create_synthesis_agent,
        input_state_type=SynthesisInputState,
        state_key="synthesis",
        error_fallback={"signal": "hold", "composite_score": 0.0},
        store=store,
    )
