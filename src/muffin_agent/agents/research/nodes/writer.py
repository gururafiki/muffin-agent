"""Writer stage: synthesise the final cited answer.

A compiled ReAct agent (no tools) added to the graph via ``add_node``, with structured
output enforcing the public ``ResearchOutput`` contract. The reranked evidence and the
task context are rendered into its first human message from state, so the agent owns
its own ``checkpoint_ns`` and its composition step is inspectable on its own.

:func:`finalize_output_node` is a pure post-step that overwrites the ``task_type`` /
``mode_used`` echo fields with the pipeline's own truth — the model is asked to repeat
them and can mislabel, and these two are metadata the caller reads.

Role: ``orchestrator`` — composition + instruction-following. Errors propagate;
``RetryPolicy`` on the node plus the model-fallback chain are the resilience layers.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain.agents.structured_output import AutoStrategy
from langchain_core.runnables import RunnableConfig

from ....model_config import ModelConfiguration
from ....utils.agent_builder import MuffinAgentBuilder
from ..schemas import WriterNodeOutput
from ..state import ResearchState, WriterAgentState

logger = logging.getLogger(__name__)


async def create_writer_agent(config: RunnableConfig):
    """Build the writer ReAct agent (no tools) as a compiled graph node."""
    model_cfg = ModelConfiguration.from_runnable_config(config)
    primary, *fallbacks = model_cfg.get_llm_for_role("orchestrator")

    return (
        MuffinAgentBuilder(primary, name="research_writer")
        .with_state_schema(WriterAgentState)
        .with_input_prompt_template("research/writer.jinja")
        .with_fallback_models(*fallbacks)
        .with_response_format(AutoStrategy(schema=WriterNodeOutput))
        .build_react_agent()
    )


def finalize_output_node(state: ResearchState) -> dict[str, Any]:
    """Pin the answer's echoed metadata to the pipeline's own values.

    ``ResearchOutput.task_type`` / ``.mode_used`` are asked of the model so they ride
    the public contract, but the pipeline already knows both — so a mislabel here is
    a silent lie to the caller. Overwrite rather than trust.
    """
    output = dict(state.get("output") or {})
    if not output:
        return {}
    output["task_type"] = state.get("task_type", "research_report")
    output["mode_used"] = state.get("mode", "balanced")
    return {"output": output}
