"""Classifier node: turn the user's query into a typed plan.

Single LLM call (collector role) that produces a ``ResearchClassification``.
The node lifts the result into flat state keys (``mode``, ``task_type``,
``sources_to_use``, ``skip_search``, ``standalone_query``) so downstream
stages — and the researcher's ``SkillFilterMiddleware`` — can read them
directly.
"""

from __future__ import annotations

import datetime as _dt
import logging
from typing import Any

from langchain.agents.structured_output import AutoStrategy
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from ....model_config import ModelConfiguration
from ....utils.agent_builder import MuffinAgentBuilder
from ...research.config import ResearchConfiguration
from ...research.schemas import ResearchClassification
from ...research.state import ResearchState

logger = logging.getLogger(__name__)


async def create_classifier_agent(
    config: RunnableConfig,
    *,
    allowed_sources: list[str],
    chat_history_text: str,
):
    """Build a lightweight ReAct agent (no tools) for classification.

    Universal middleware (retries, fallback models) applies; structured
    output is enforced via ``AutoStrategy``.
    """
    model_cfg = ModelConfiguration.from_runnable_config(config)
    primary, *fallbacks = model_cfg.get_llm_for_role("collector")

    today = _dt.date.today().isoformat()

    builder = (
        MuffinAgentBuilder(primary, name="research_classifier")
        .with_system_prompt_template(
            "research/classifier.jinja",
            allowed_sources=allowed_sources,
            chat_history=chat_history_text,
            today=today,
        )
        .with_fallback_models(*fallbacks)
        .with_response_format(AutoStrategy(schema=ResearchClassification))
    )
    return builder.build_react_agent()


def _render_chat_history(messages: list[Any] | None) -> str:
    """Render the last ~6 messages as ``User:``/``Assistant:`` blocks."""
    if not messages:
        return "(no prior conversation)"
    rendered: list[str] = []
    for msg in messages[-6:]:
        role = getattr(msg, "type", None) or getattr(msg, "role", None) or "user"
        role_str = str(role)
        content = getattr(msg, "content", str(msg))
        label = {"human": "User", "ai": "Assistant"}.get(role_str, role_str.title())
        rendered.append(f"{label}: {content}")
    return "\n".join(rendered)


async def classifier_node(
    state: ResearchState,
    config: RunnableConfig,
    *,
    extra_sources: list[str] | None = None,
) -> dict[str, Any]:
    """Classify the query and lift the result into flat state keys.

    Honours ``mode_override`` / ``task_type_override`` from the caller.
    Defensively intersects ``sources_to_use`` with ``allowed_sources``
    so a wandering classifier can't enable a source the caller hasn't
    permitted.
    """
    research_cfg = ResearchConfiguration.from_runnable_config(config)

    allowed_sources = state.get("allowed_sources") or [
        *research_cfg.research_default_sources,
        *(extra_sources or []),
    ]
    # Deduplicate while preserving order.
    seen: set[str] = set()
    deduped: list[str] = []
    for src in allowed_sources:
        if src not in seen:
            seen.add(src)
            deduped.append(src)
    allowed_sources = deduped

    chat_history_text = _render_chat_history(state.get("chat_history"))

    try:
        agent = await create_classifier_agent(
            config,
            allowed_sources=allowed_sources,
            chat_history_text=chat_history_text,
        )
        result = await agent.ainvoke(
            {"messages": [HumanMessage(content=state["query"])]},
            config=config,
        )
        structured = (
            result.get("structured_response") if isinstance(result, dict) else None
        )
    except Exception:
        logger.exception("classifier_node failed; falling back to defaults")
        structured = None

    if structured is None:
        # Fallback: assume balanced web research on the raw query.
        return {
            "standalone_query": state["query"],
            "task_type": state.get("task_type_override") or "research_report",
            "mode": state.get("mode_override") or research_cfg.research_default_mode,
            "sources_to_use": allowed_sources or ["web"],
            "skip_search": False,
            "classification": {
                "error": "classifier_did_not_produce_structured_output",
                "fallback": True,
            },
        }

    payload = structured.model_dump()
    intersected = [s for s in payload["sources_to_use"] if s in allowed_sources]
    if not intersected and not payload["skip_search"]:
        intersected = allowed_sources or ["web"]

    mode = state.get("mode_override") or payload["mode_hint"]
    task_type = state.get("task_type_override") or payload["task_type"]

    return {
        "standalone_query": payload["standalone_query"],
        "task_type": task_type,
        "mode": mode,
        "sources_to_use": intersected,
        "skip_search": payload["skip_search"],
        "classification": payload,
    }
