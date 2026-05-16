"""Tests for the classifier node."""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.runnables import RunnableConfig

from muffin_agent.agents.research.nodes import classifier as classifier_module
from muffin_agent.agents.research.schemas import ResearchClassification


def _patch_classifier_agent(
    monkeypatch: Any, structured: ResearchClassification | None
):
    """Replace ``create_classifier_agent`` with a stub returning *structured*.

    The stub returns an awaitable agent object whose ``ainvoke`` returns
    ``{"structured_response": structured}`` (or ``{"structured_response": None}``).
    """

    class _StubAgent:
        async def ainvoke(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
            return {"structured_response": structured}

    async def _stub_create(*args: Any, **kwargs: Any) -> _StubAgent:
        return _StubAgent()

    monkeypatch.setattr(classifier_module, "create_classifier_agent", _stub_create)


@pytest.mark.unit
@pytest.mark.asyncio
class TestClassifierNode:
    async def test_lifts_classification_into_flat_keys(
        self,
        monkeypatch: Any,
        sample_classification: ResearchClassification,
    ):
        _patch_classifier_agent(monkeypatch, sample_classification)
        result = await classifier_module.classifier_node(
            {"query": "How does pgvector indexing work?"},
            RunnableConfig(configurable={}),
        )
        assert result["standalone_query"] == "How does pgvector indexing work?"
        assert result["task_type"] == "how_to"
        assert result["mode"] == "balanced"
        assert result["sources_to_use"] == ["web"]
        assert result["skip_search"] is False
        assert "classification" in result

    async def test_mode_override_takes_precedence(
        self,
        monkeypatch: Any,
        sample_classification: ResearchClassification,
    ):
        _patch_classifier_agent(monkeypatch, sample_classification)
        result = await classifier_module.classifier_node(
            {"query": "How does pgvector indexing work?", "mode_override": "quality"},
            RunnableConfig(configurable={}),
        )
        assert result["mode"] == "quality"

    async def test_task_type_override_takes_precedence(
        self,
        monkeypatch: Any,
        sample_classification: ResearchClassification,
    ):
        _patch_classifier_agent(monkeypatch, sample_classification)
        result = await classifier_module.classifier_node(
            {
                "query": "How does pgvector indexing work?",
                "task_type_override": "comparison",
            },
            RunnableConfig(configurable={}),
        )
        assert result["task_type"] == "comparison"

    async def test_sources_intersected_with_allowed(
        self,
        monkeypatch: Any,
    ):
        # Classifier returns "academic" but the caller only allowed "web".
        wandering = ResearchClassification(
            standalone_query="x",
            task_type="research_report",
            mode_hint="balanced",
            sources_to_use=["web", "academic"],
            skip_search=False,
        )
        _patch_classifier_agent(monkeypatch, wandering)

        result = await classifier_module.classifier_node(
            {"query": "x", "allowed_sources": ["web"]},
            RunnableConfig(configurable={}),
        )
        assert result["sources_to_use"] == ["web"]

    async def test_extra_sources_appended_to_allowed_default(
        self,
        monkeypatch: Any,
    ):
        agreeable = ResearchClassification(
            standalone_query="x",
            task_type="research_report",
            mode_hint="balanced",
            sources_to_use=["academic"],
            skip_search=False,
        )
        _patch_classifier_agent(monkeypatch, agreeable)

        result = await classifier_module.classifier_node(
            {"query": "x"},
            RunnableConfig(configurable={}),
            extra_sources=["academic"],
        )
        assert "academic" in result["sources_to_use"]

    async def test_fallback_when_no_structured_response(self, monkeypatch: Any):
        _patch_classifier_agent(monkeypatch, None)
        result = await classifier_module.classifier_node(
            {"query": "Anything"},
            RunnableConfig(configurable={}),
        )
        assert result["standalone_query"] == "Anything"
        assert result["task_type"] == "research_report"
        assert result["skip_search"] is False
        assert result["classification"]["fallback"] is True


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
