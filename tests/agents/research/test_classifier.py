"""Tests for the classifier stage's pure nodes.

The classifier itself is now a compiled agent added via ``add_node``, so the logic
worth testing lives in the two pure nodes around it: ``prepare_node`` (normalise the
caller's input) and ``lift_classification_node`` (flatten the agent's output, apply
overrides, intersect sources). Both are plain functions — no LLM stubbing needed.
"""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.runnables import RunnableConfig

from muffin_agent.agents.research.nodes import classifier as classifier_module
from muffin_agent.agents.research.schemas import ResearchClassification

_CFG = RunnableConfig(configurable={})


def _classified(**overrides: Any) -> dict[str, Any]:
    """The ``classification`` payload the classifier agent unpacks into state."""
    base = ResearchClassification(
        standalone_query="How does pgvector indexing work?",
        task_type="how_to",
        mode_hint="balanced",
        sources_to_use=["web"],
        skip_search=False,
        rationale="how-to query about a specific technical mechanism",
    ).model_dump()
    base.update(overrides)
    return base


@pytest.mark.unit
class TestLiftClassification:
    def test_lifts_classification_into_flat_keys(self):
        result = classifier_module.lift_classification_node(
            {
                "query": "How does pgvector indexing work?",
                "allowed_sources": ["web"],
                "classification": _classified(),
            },
            _CFG,
        )
        assert result["standalone_query"] == "How does pgvector indexing work?"
        assert result["task_type"] == "how_to"
        assert result["mode"] == "balanced"
        assert result["sources_to_use"] == ["web"]
        assert result["skip_search"] is False

    def test_mode_override_takes_precedence(self):
        result = classifier_module.lift_classification_node(
            {
                "allowed_sources": ["web"],
                "classification": _classified(),
                "mode_override": "quality",
            },
            _CFG,
        )
        assert result["mode"] == "quality"

    def test_task_type_override_takes_precedence(self):
        result = classifier_module.lift_classification_node(
            {
                "allowed_sources": ["web"],
                "classification": _classified(),
                "task_type_override": "comparison",
            },
            _CFG,
        )
        assert result["task_type"] == "comparison"

    def test_sources_intersected_with_allowed(self):
        """A wandering classifier cannot enable a source the caller never permitted."""
        result = classifier_module.lift_classification_node(
            {
                "allowed_sources": ["web"],
                "classification": _classified(sources_to_use=["web", "academic"]),
            },
            _CFG,
        )
        assert result["sources_to_use"] == ["web"]

    def test_empty_intersection_falls_back_to_allowed(self):
        """Never leave the researcher with zero sources on a searching run."""
        result = classifier_module.lift_classification_node(
            {
                "allowed_sources": ["web"],
                "classification": _classified(sources_to_use=["academic"]),
            },
            _CFG,
        )
        assert result["sources_to_use"] == ["web"]

    def test_skip_search_keeps_empty_sources(self):
        result = classifier_module.lift_classification_node(
            {
                "allowed_sources": ["web"],
                "classification": _classified(sources_to_use=[], skip_search=True),
            },
            _CFG,
        )
        assert result["skip_search"] is True
        assert result["sources_to_use"] == []


@pytest.mark.unit
class TestPrepareNode:
    def test_defaults_and_dedupes_allowed_sources(self):
        result = classifier_module.prepare_node(
            {"query": "x", "allowed_sources": ["web", "web", "academic"]}, _CFG
        )
        assert result["allowed_sources"] == ["web", "academic"]

    def test_caller_sources_win_over_config_defaults(self):
        result = classifier_module.prepare_node(
            {"query": "x", "allowed_sources": ["academic"]}, _CFG
        )
        assert result["allowed_sources"] == ["academic"]

    def test_supplies_today_and_chat_history_text(self):
        result = classifier_module.prepare_node({"query": "x"}, _CFG)
        assert result["today"].count("-") == 2  # ISO date
        assert result["chat_history_text"].startswith("(no")


@pytest.mark.unit
class TestRenderChatHistory:
    def test_empty_history(self):
        assert classifier_module._render_chat_history(None).startswith("(no")
        assert classifier_module._render_chat_history([]).startswith("(no")

    def test_renders_known_roles(self):
        class _Msg:
            def __init__(self, role: str, content: str) -> None:
                self.type = role
                self.content = content

        rendered = classifier_module._render_chat_history(
            [_Msg("human", "hi"), _Msg("ai", "hello")]
        )
        assert "User: hi" in rendered
        assert "Assistant: hello" in rendered
