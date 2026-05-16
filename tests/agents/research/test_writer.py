"""Tests for the writer node."""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.runnables import RunnableConfig

from muffin_agent.agents.research.nodes import writer as writer_module
from muffin_agent.agents.research.schemas import ResearchOutput


def _patch_writer_agent(monkeypatch: Any, structured: ResearchOutput | None):
    class _StubAgent:
        async def ainvoke(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
            return {"structured_response": structured}

    async def _stub_create(*args: Any, **kwargs: Any) -> _StubAgent:
        return _StubAgent()

    monkeypatch.setattr(writer_module, "create_writer_agent", _stub_create)


@pytest.mark.unit
@pytest.mark.asyncio
class TestWriterNode:
    async def test_returns_validated_output(
        self,
        monkeypatch: Any,
        sample_research_output: ResearchOutput,
    ):
        _patch_writer_agent(monkeypatch, sample_research_output)
        result = await writer_module.writer_node(
            {
                "query": "How does pgvector indexing work?",
                "task_type": "how_to",
                "mode": "balanced",
                "reranked_evidence": [],
                "skip_search": False,
            },
            RunnableConfig(configurable={}),
        )
        output = result["output"]
        # Re-validation must pass.
        ResearchOutput.model_validate(output)
        assert output["task_type"] == "how_to"
        assert output["mode_used"] == "balanced"

    async def test_fallback_when_no_structured_response(self, monkeypatch: Any):
        _patch_writer_agent(monkeypatch, None)
        result = await writer_module.writer_node(
            {"query": "Anything", "task_type": "summary", "mode": "speed"},
            RunnableConfig(configurable={}),
        )
        output = result["output"]
        ResearchOutput.model_validate(output)
        assert output["confidence"] == 0.0
        assert output["task_type"] == "summary"
        assert output["mode_used"] == "speed"

    async def test_task_type_and_mode_echoed_from_state(
        self,
        monkeypatch: Any,
        sample_research_output: ResearchOutput,
    ):
        # Writer LLM returns task_type/mode_used that diverge from state;
        # node must defensively echo state values.
        result_obj = sample_research_output.model_copy(
            update={"task_type": "wrong", "mode_used": "wrong"}
        )
        _patch_writer_agent(monkeypatch, result_obj)
        result = await writer_module.writer_node(
            {"query": "x", "task_type": "comparison", "mode": "quality"},
            RunnableConfig(configurable={}),
        )
        assert result["output"]["task_type"] == "comparison"
        assert result["output"]["mode_used"] == "quality"
