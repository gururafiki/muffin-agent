"""Tests for investment graph builder utilities."""

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.memory import InMemoryStore

from muffin_agent.agents.equity_screening import build_equity_screening_graph
from muffin_agent.agents.investment_analysis import build_investment_analysis_graph


@pytest.mark.unit
class TestBuildInvestmentAnalysisGraph:
    """Tests for build_investment_analysis_graph."""

    def test_compiles_without_checkpointer(self):
        graph = build_investment_analysis_graph()
        assert isinstance(graph, CompiledStateGraph)

    def test_compiles_with_checkpointer(self):
        graph = build_investment_analysis_graph(checkpointer=InMemorySaver())
        assert isinstance(graph, CompiledStateGraph)

    def test_default_checkpointer_is_none(self):
        graph = build_investment_analysis_graph()
        assert graph.checkpointer is None

    def test_checkpointer_is_set_when_provided(self):
        saver = InMemorySaver()
        graph = build_investment_analysis_graph(checkpointer=saver)
        assert graph.checkpointer is saver

    def test_compiles_with_store(self):
        store = InMemoryStore()
        graph = build_investment_analysis_graph(store=store)
        assert isinstance(graph, CompiledStateGraph)
        assert graph.store is store

    def test_default_store_is_none(self):
        graph = build_investment_analysis_graph()
        assert graph.store is None


@pytest.mark.unit
class TestBuildEquityScreeningGraph:
    """Tests for build_equity_screening_graph."""

    def test_compiles_without_checkpointer(self):
        graph = build_equity_screening_graph()
        assert isinstance(graph, CompiledStateGraph)

    def test_compiles_with_checkpointer(self):
        graph = build_equity_screening_graph(checkpointer=InMemorySaver())
        assert isinstance(graph, CompiledStateGraph)

    def test_default_checkpointer_is_none(self):
        graph = build_equity_screening_graph()
        assert graph.checkpointer is None

    def test_checkpointer_is_set_when_provided(self):
        saver = InMemorySaver()
        graph = build_equity_screening_graph(checkpointer=saver)
        assert graph.checkpointer is saver

    def test_compiles_with_store(self):
        store = InMemoryStore()
        graph = build_equity_screening_graph(store=store)
        assert isinstance(graph, CompiledStateGraph)
        assert graph.store is store

    def test_default_store_is_none(self):
        graph = build_equity_screening_graph()
        assert graph.store is None
