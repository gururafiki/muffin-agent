"""End-to-end smoke tests for the research graph.

Every LLM stage is now a compiled agent added via ``add_node``, so the stubs here are
``RunnableLambda``s standing in for those compiled agents. They return the state
updates a real agent emits *after* ``_StructuredResponseToStateMiddleware`` unpacks
its structured response — which is exactly the contract the graph wiring depends on.
"""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.memory import InMemoryStore

from muffin_agent.agents.research import graph as graph_module
from muffin_agent.agents.research.graph import build_research_graph
from muffin_agent.agents.research.schemas import (
    ResearchClassification,
    ResearcherNodeOutput,
    ResearchOutput,
)
from muffin_agent.agents.research.state import RESEARCH_MODES

_CFG = RunnableConfig(configurable={})


def _patch_all_agents(
    monkeypatch: Any,
    *,
    classification: ResearchClassification,
    findings: ResearcherNodeOutput,
    output: ResearchOutput,
) -> dict[str, list[str]]:
    """Stub every compiled agent node; return a record of which ones ran.

    Patch on ``graph_module``, NOT on the node modules: ``graph.py`` does
    ``from .nodes import create_classifier_agent``, which binds the function into
    its own namespace at import time, so patching the node module's attribute
    silently has no effect and the REAL agent (and a real LLM call) runs.
    """
    ran: dict[str, list[str]] = {"researchers": []}

    async def _classifier(_state: dict[str, Any]) -> dict[str, Any]:
        return {"classification": classification.model_dump()}

    def _make_researcher(mode: str) -> RunnableLambda:
        async def _researcher(_state: dict[str, Any]) -> dict[str, Any]:
            ran["researchers"].append(mode)
            payload = findings.model_dump()
            return {"evidence": payload["evidence"], "notes": payload["notes"]}

        return RunnableLambda(_researcher)

    async def _writer(_state: dict[str, Any]) -> dict[str, Any]:
        return {"output": output.model_dump()}

    async def _classifier_factory(*_a: Any, **_kw: Any) -> RunnableLambda:
        return RunnableLambda(_classifier)

    async def _researchers_factory(*_a: Any, **_kw: Any) -> dict[str, RunnableLambda]:
        return {mode: _make_researcher(mode) for mode in RESEARCH_MODES}

    async def _writer_factory(*_a: Any, **_kw: Any) -> RunnableLambda:
        return RunnableLambda(_writer)

    monkeypatch.setattr(graph_module, "create_classifier_agent", _classifier_factory)
    monkeypatch.setattr(graph_module, "build_researchers_by_mode", _researchers_factory)
    monkeypatch.setattr(graph_module, "create_writer_agent", _writer_factory)
    return ran


@pytest.fixture
def stub_agents(
    monkeypatch: Any,
    sample_classification: ResearchClassification,
    sample_evidence_findings: ResearcherNodeOutput,
    sample_research_output: ResearchOutput,
) -> dict[str, list[str]]:
    return _patch_all_agents(
        monkeypatch,
        classification=sample_classification,
        findings=sample_evidence_findings,
        output=sample_research_output,
    )


@pytest.mark.unit
@pytest.mark.asyncio
class TestBuildGraph:
    async def test_compiles_without_checkpointer_or_store(self, stub_agents):
        assert isinstance(await build_research_graph(_CFG), CompiledStateGraph)

    async def test_compiles_with_checkpointer(self, stub_agents):
        g = await build_research_graph(_CFG, checkpointer=MemorySaver())
        assert isinstance(g, CompiledStateGraph)

    async def test_compiles_with_store(self, stub_agents):
        g = await build_research_graph(_CFG, store=InMemoryStore())
        assert isinstance(g, CompiledStateGraph)

    async def test_expected_nodes(self, stub_agents):
        """Every LLM stage is its own node — that is what gives it a namespace."""
        g = await build_research_graph(_CFG)
        names = set(g.nodes.keys())
        assert {
            "prepare",
            "classifier",
            "lift_classification",
            "researcher_speed",
            "researcher_balanced",
            "researcher_quality",
            "rerank",
            "writer",
            "finalize",
        } <= names


@pytest.mark.unit
class TestResearcherRouting:
    @pytest.mark.parametrize(
        "mode,expected",
        [
            ("speed", "researcher_speed"),
            ("balanced", "researcher_balanced"),
            ("quality", "researcher_quality"),
            (None, "researcher_balanced"),
            ("nonsense", "researcher_balanced"),
        ],
    )
    def test_routes_to_the_matching_budget(self, mode, expected):
        state = {"mode": mode} if mode is not None else {}
        assert graph_module._route_researcher(state) == expected

    def test_skip_search_bypasses_every_researcher(self):
        route = graph_module._route_after_classification({"skip_search": True})
        assert route == "writer"


@pytest.mark.unit
@pytest.mark.asyncio
class TestEndToEnd:
    async def test_default_path_runs_all_nodes(
        self,
        stub_agents: dict[str, list[str]],
        mock_embedder: dict[str, Any],  # noqa: ARG002 — patches OpenAIEmbeddings
    ):
        g = await build_research_graph(_CFG)
        result = await g.ainvoke(
            {"query": "How does pgvector indexing work?"}, config=_CFG
        )
        # Output is the writer's structured response, schema-valid.
        ResearchOutput.model_validate(result["output"])
        # Researcher produced evidence, rerank trimmed to the relevant subset.
        assert len(result["evidence"]) == 3
        assert len(result["reranked_evidence"]) == 2
        # Flat classification keys are present.
        assert result["task_type"] == "how_to"
        assert result["mode"] == "balanced"
        # Exactly the budget-matched researcher ran.
        assert stub_agents["researchers"] == ["balanced"]

    async def test_mode_override_selects_a_different_researcher(
        self,
        stub_agents: dict[str, list[str]],
        mock_embedder: dict[str, Any],  # noqa: ARG002
    ):
        g = await build_research_graph(_CFG)
        result = await g.ainvoke(
            {"query": "How does pgvector indexing work?", "mode_override": "quality"},
            config=_CFG,
        )
        assert result["mode"] == "quality"
        assert stub_agents["researchers"] == ["quality"]

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
        ran = _patch_all_agents(
            monkeypatch,
            classification=skip,
            findings=ResearcherNodeOutput(evidence=[]),
            output=sample_research_output,
        )

        g = await build_research_graph(_CFG)
        result = await g.ainvoke({"query": "What is 2+2?"}, config=_CFG)
        assert result["skip_search"] is True
        ResearchOutput.model_validate(result["output"])
        assert ran["researchers"] == []
        assert not result.get("reranked_evidence")
