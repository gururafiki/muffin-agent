"""Writer node: synthesise the final cited answer.

Plain ReAct agent (no tools) with structured output enforcing the
public ``ResearchOutput`` contract.  Receives reranked evidence in
the rendered system prompt.

Role: ``orchestrator`` — composition + instruction-following.
"""

from __future__ import annotations

import datetime as _dt
import logging
from typing import Any

from langchain.agents.structured_output import AutoStrategy
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.store.base import BaseStore

from ....model_config import ModelConfiguration
from ....utils.agent_builder import MuffinAgentBuilder
from ..schemas import ResearchOutput
from ..state import ResearchState

logger = logging.getLogger(__name__)


_EMPTY_FALLBACK_ANSWER = (
    "Sorry — I wasn't able to compose an answer for this query. "
    "Please retry with a more specific question."
)


def _fallback_output(state: ResearchState) -> dict[str, Any]:
    """Construct a schema-valid ResearchOutput when the writer fails."""
    fallback = ResearchOutput(
        answer_markdown=_EMPTY_FALLBACK_ANSWER,
        key_findings=[],
        sources=[],
        confidence=0.0,
        missing_information=["writer_node failed to produce structured output"],
        suggested_followups=[],
        task_type=state.get("task_type", "research_report"),
        mode_used=state.get("mode", "balanced"),
    )
    return fallback.model_dump()


async def create_writer_agent(
    config: RunnableConfig,
    *,
    query: str,
    task_type: str,
    mode: str,
    evidence: list[dict[str, Any]],
    skip_search_flag: bool,
    system_instructions: str | None,
):
    """Build the writer ReAct agent (no tools)."""
    model_cfg = ModelConfiguration.from_runnable_config(config)
    primary, *fallbacks = model_cfg.get_llm_for_role("orchestrator")

    today = _dt.date.today().isoformat()

    builder = (
        MuffinAgentBuilder(primary, name="research_writer")
        .with_system_prompt_template(
            "research/writer.jinja",
            today=today,
            query=query,
            task_type=task_type,
            mode=mode,
            evidence=evidence,
            skip_search_flag=skip_search_flag,
            system_instructions=system_instructions or "",
        )
        .with_fallback_models(*fallbacks)
        .with_response_format(AutoStrategy(schema=ResearchOutput))
    )
    return builder.build_react_agent()


async def writer_node(
    state: ResearchState,
    config: RunnableConfig,
    *,
    store: BaseStore | None = None,  # noqa: ARG001 — kept for graph-wiring symmetry
) -> dict[str, Any]:
    """Compose the final cited answer."""
    query = state.get("standalone_query") or state.get("query", "")
    task_type = state.get("task_type", "research_report")
    mode = state.get("mode", "balanced")
    evidence = state.get("reranked_evidence") or []
    skip_search_flag = bool(state.get("skip_search"))

    try:
        agent = await create_writer_agent(
            config,
            query=query,
            task_type=task_type,
            mode=mode,
            evidence=evidence,
            skip_search_flag=skip_search_flag,
            system_instructions=state.get("system_instructions"),
        )
        # The system prompt already carries the full context; the user
        # message is just a kick-off cue.
        result = await agent.ainvoke(
            {"messages": [HumanMessage(content=query)]},
            config=config,
        )
        structured = (
            result.get("structured_response") if isinstance(result, dict) else None
        )
    except Exception:
        logger.exception("writer_node failed; returning fallback output")
        return {"output": _fallback_output(state)}

    if structured is None:
        logger.warning("writer_node: no structured_response on agent result")
        return {"output": _fallback_output(state)}

    payload = structured.model_dump()
    # Defensive echo: ensure task_type / mode_used match the state
    # even if the LLM mislabelled them.
    payload["task_type"] = task_type
    payload["mode_used"] = mode
    return {"output": payload}
