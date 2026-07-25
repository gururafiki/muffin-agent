"""End-to-end smoke tests for the research graph."""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.memory import InMemoryStore

from muffin_agent.agents.research.graph import build_research_graph, graph
from muffin_agent.agents.research.nodes import classifier as classifier_module
from muffin_agent.agents.research.nodes import researcher as researcher_module
from muffin_agent.agents.research.nodes import writer as writer_module
from muffin_agent.agents.research.schemas import (
    ResearchClassification,
    ResearchEvidenceFindings,
    ResearchOutput,
)

#: One tool-execution record the researcher stub emits, mimicking
#: ``AgentCaptureMiddleware`` output so the node's ``tool_runs`` forwarding + the
#: ``ResearchState`` channel are exercised end-to-end.
_STUB_TOOL_RUN = {"tool": "firecrawl_search", "agent": "deep-research", "status": "ok"}

#: One sub-agent-tree node the researcher stub emits, mimicking
#: ``AgentCaptureMiddleware`` output so the node's ``subagent_tree`` forwarding +
#: the ``ResearchState`` channel are exercised end-to-end (Task 5 propagation).
_STUB_SUBAGENT_TREE = {
    "__root__": {"id": "__root__", "name": "deep-research", "kind": "subgraph"}
}


@pytest.mark.unit
class TestBuildGraph:
    def test_module_level_graph_is_compiled(self):
        assert isinstance(graph, CompiledStateGraph)

    def test_compiles_without_checkpointer_or_store(self):
        g = build_research_graph()
        assert isinstance(g, CompiledStateGraph)

    def test_compiles_with_checkpointer(self):
        g = build_research_graph(checkpointer=MemorySaver())
        assert isinstance(g, CompiledStateGraph)

    def test_compiles_with_store(self):
        g = build_research_graph(store=InMemoryStore())
        assert isinstance(g, CompiledStateGraph)

    def test_expected_nodes(self):
        g = build_research_graph()
        names = set(g.nodes.keys())
        assert {"classifier", "researcher", "rerank", "writer"} <= names


def _patch_all_nodes(
    monkeypatch: Any,
    *,
    classification: ResearchClassification,
    findings: ResearchEvidenceFindings,
    output: ResearchOutput,
):
    """Replace every LLM-backed node's agent with stubs."""

    class _ClassifierStub:
        async def ainvoke(self, *_a: Any, **_kw: Any) -> dict[str, Any]:
            return {"structured_response": classification}

    class _ResearcherStub:
        async def ainvoke(self, *_a: Any, **_kw: Any) -> dict[str, Any]:
            # Mimic the deep agent's AgentCaptureMiddleware output so the node's
            # tool_runs/subagent_tree forwarding + the ResearchState channel are
            # exercised.
            return {
                "structured_response": findings,
                "tool_runs": [_STUB_TOOL_RUN],
                "subagent_tree": _STUB_SUBAGENT_TREE,
            }

    class _WriterStub:
        async def ainvoke(self, *_a: Any, **_kw: Any) -> dict[str, Any]:
            return {"structured_response": output}

    async def _classifier_factory(*_a: Any, **_kw: Any) -> _ClassifierStub:
        return _ClassifierStub()

    async def _researcher_factory(*_a: Any, **_kw: Any) -> _ResearcherStub:
        return _ResearcherStub()

    async def _writer_factory(*_a: Any, **_kw: Any) -> _WriterStub:
        return _WriterStub()

    monkeypatch.setattr(
        classifier_module, "create_classifier_agent", _classifier_factory
    )
    monkeypatch.setattr(
        researcher_module, "create_researcher_agent", _researcher_factory
    )
    monkeypatch.setattr(writer_module, "create_writer_agent", _writer_factory)


@pytest.mark.unit
@pytest.mark.asyncio
class TestEndToEnd:
    async def test_default_path_runs_all_nodes(
        self,
        monkeypatch: Any,
        sample_classification: ResearchClassification,
        sample_evidence_findings: ResearchEvidenceFindings,
        sample_research_output: ResearchOutput,
        mock_embedder: dict[str, Any],  # noqa: ARG002 — patches OpenAIEmbeddings
    ):
        _patch_all_nodes(
            monkeypatch,
            classification=sample_classification,
            findings=sample_evidence_findings,
            output=sample_research_output,
        )

        g = build_research_graph()
        result = await g.ainvoke(
            {"query": "How does pgvector indexing work?"},
            config=RunnableConfig(configurable={}),
        )
        # Output is the writer's structured response, schema-valid.
        ResearchOutput.model_validate(result["output"])
        # Researcher produced evidence, rerank trimmed to relevant subset.
        assert len(result["evidence"]) == 3
        # Rerank drops the unrelated python chunk.
        assert len(result["reranked_evidence"]) == 2
        # Flat classification keys are present.
        assert result["task_type"] == "how_to"
        assert result["mode"] == "balanced"
        # The researcher's captured tool_runs surface for the "Tool execution" panel.
        assert result["tool_runs"] == [_STUB_TOOL_RUN]
        # Same propagation proof for the sub-agent execution tree (Task 5).
        assert result["subagent_tree"] == _STUB_SUBAGENT_TREE

    async def test_skip_search_bypasses_researcher_and_rerank(
        self,
        monkeypatch: Any,
        sample_research_output: ResearchOutput,
    ):
        skip = ResearchClassification(
            standalone_query="What is 2+2?",
            task_type="factual_qa",
            mode_hint="speed",
            sources_to_use=[],
            skip_search=True,
            rationale="trivial arithmetic",
        )
        # Researcher/embeddings stubs should never run; provide harmless ones.
        _patch_all_nodes(
            monkeypatch,
            classification=skip,
            findings=ResearchEvidenceFindings(evidence_chunks=[]),
            output=sample_research_output,
        )

        g = build_research_graph()
        result = await g.ainvoke(
            {"query": "What is 2+2?"},
            config=RunnableConfig(configurable={}),
        )
        assert result["skip_search"] is True
        ResearchOutput.model_validate(result["output"])
        # Rerank was never called on this path.
        assert "reranked_evidence" not in result or result["reranked_evidence"] == []
