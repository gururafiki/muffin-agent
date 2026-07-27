"""Classifier stage: turn the user's query into a typed plan.

Three pieces, split so each has exactly one job:

* :func:`prepare_node` — pure Python. Normalises ``allowed_sources`` (defaults +
  dedupe) and renders the chat history into text, so the classifier's input prompt
  reads them straight off state.
* :func:`create_classifier_agent` — a compiled ReAct agent (no tools) added to the
  graph via ``add_node``. Because it is a real graph node it gets its own
  ``checkpoint_ns``, so its transcript is inspectable independently of the pipeline.
* :func:`lift_classification_node` — pure Python. Flattens ``classification`` into
  the flat keys downstream stages (and the researcher's ``SkillFilterMiddleware``)
  read, applying the caller's overrides and intersecting ``sources_to_use`` with
  ``allowed_sources`` so a wandering classifier can't enable a source the caller
  never permitted.

This mirrors ``criteria_analysis``'s ``ticker_classification`` → ``lift_classification``
pair. Errors propagate (``RetryPolicy`` on the node + the model-fallback chain);
there is no fallback dict.
"""

from __future__ import annotations

import datetime as _dt
import logging
from typing import Any

from langchain.agents.structured_output import AutoStrategy
from langchain_core.runnables import RunnableConfig

from ....model_config import ModelConfiguration
from ....utils.agent_builder import MuffinAgentBuilder
from ..config import ResearchConfiguration
from ..schemas import ClassifierNodeOutput
from ..state import ClassifierAgentState, ResearchState

logger = logging.getLogger(__name__)


async def create_classifier_agent(config: RunnableConfig):
    """Build the classifier ReAct agent (no tools) as a compiled graph node."""
    model_cfg = ModelConfiguration.from_runnable_config(config)
    primary, *fallbacks = model_cfg.get_llm_for_role("collector")

    return (
        MuffinAgentBuilder(primary, name="research_classifier")
        .with_state_schema(ClassifierAgentState)
        .with_input_prompt_template("research/classifier.jinja")
        .with_fallback_models(*fallbacks)
        .with_response_format(AutoStrategy(schema=ClassifierNodeOutput))
        .build_react_agent()
    )


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


def prepare_node(state: ResearchState, config: RunnableConfig) -> dict[str, Any]:
    """Normalise caller input into the shape the classifier prompt expects."""
    research_cfg = ResearchConfiguration.from_runnable_config(config)

    allowed_sources = state.get("allowed_sources") or list(
        research_cfg.research_default_sources
    )
    seen: set[str] = set()
    deduped: list[str] = []
    for src in allowed_sources:
        if src not in seen:
            seen.add(src)
            deduped.append(src)

    return {
        "allowed_sources": deduped or ["web"],
        "chat_history_text": _render_chat_history(state.get("chat_history")),
        # Resolved here rather than at import time so a long-lived server process
        # doesn't keep serving the date it booted on.
        "today": _dt.date.today().isoformat(),
    }


def lift_classification_node(
    state: ResearchState, config: RunnableConfig
) -> dict[str, Any]:
    """Flatten ``classification`` into the flat keys downstream stages read."""
    research_cfg = ResearchConfiguration.from_runnable_config(config)
    payload = dict(state.get("classification") or {})
    allowed_sources = state.get("allowed_sources") or ["web"]

    skip_search = bool(payload.get("skip_search", False))
    intersected = [s for s in payload.get("sources_to_use", []) if s in allowed_sources]
    if not intersected and not skip_search:
        intersected = allowed_sources

    mode = state.get("mode_override") or payload.get("mode_hint")
    task_type = state.get("task_type_override") or payload.get("task_type")

    return {
        "standalone_query": payload.get("standalone_query") or state.get("query", ""),
        "task_type": task_type or "research_report",
        "mode": mode or research_cfg.research_default_mode,
        "sources_to_use": intersected,
        "skip_search": skip_search,
    }
