"""Tests for the investment graph builders.

Both builders are async now: every stage is a compiled deep agent built at
graph-construction time rather than per request. The MCP client is mocked at its
construction site in ``data_collection.utils`` — patching ``get_tools`` itself would
NOT work, because each collector module binds that name into its own namespace at
import time.
"""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.memory import InMemoryStore

from muffin_agent.agents.equity_screening import build_equity_screening_graph
from muffin_agent.agents.investment_analysis import build_investment_analysis_graph

_CFG: dict[str, Any] = {"configurable": {}}


@pytest.fixture(autouse=True)
def _no_mcp():
    """Stub the MCP client so agent construction never touches the network."""
    client = MagicMock()
    client.get_tools = AsyncMock(return_value=[])
    with patch(
        "muffin_agent.agents.data_collection.utils.MultiServerMCPClient",
        MagicMock(return_value=client),
    ):
        yield


@pytest.mark.unit
@pytest.mark.asyncio
class TestBuildInvestmentAnalysisGraph:
    """Tests for build_investment_analysis_graph."""

    async def test_compiles_without_checkpointer(self):
        graph = await build_investment_analysis_graph(_CFG)
        assert isinstance(graph, CompiledStateGraph)

    async def test_compiles_with_checkpointer(self):
        graph = await build_investment_analysis_graph(
            _CFG, checkpointer=InMemorySaver()
        )
        assert isinstance(graph, CompiledStateGraph)

    async def test_default_checkpointer_is_none(self):
        graph = await build_investment_analysis_graph(_CFG)
        assert graph.checkpointer is None

    async def test_checkpointer_is_set_when_provided(self):
        saver = InMemorySaver()
        graph = await build_investment_analysis_graph(_CFG, checkpointer=saver)
        assert graph.checkpointer is saver

    async def test_compiles_with_store(self):
        store = InMemoryStore()
        graph = await build_investment_analysis_graph(_CFG, store=store)
        assert isinstance(graph, CompiledStateGraph)
        assert graph.store is store

    async def test_default_store_is_none(self):
        graph = await build_investment_analysis_graph(_CFG)
        assert graph.store is None

    async def test_every_stage_is_its_own_node(self):
        """Each stage must be a real node — that is what gives it a namespace."""
        graph = await build_investment_analysis_graph(_CFG)
        assert {
            "market_regime",
            "sector_analysis",
            "company_analysis",
            "forecasting",
            "risk_assessment",
            "valuation",
            "thesis_synthesis",
        } <= set(graph.nodes)


@pytest.mark.unit
@pytest.mark.asyncio
class TestBuildEquityScreeningGraph:
    """Tests for build_equity_screening_graph."""

    async def test_compiles_without_checkpointer(self):
        graph = await build_equity_screening_graph(_CFG)
        assert isinstance(graph, CompiledStateGraph)

    async def test_compiles_with_checkpointer(self):
        graph = await build_equity_screening_graph(_CFG, checkpointer=InMemorySaver())
        assert isinstance(graph, CompiledStateGraph)

    async def test_default_checkpointer_is_none(self):
        graph = await build_equity_screening_graph(_CFG)
        assert graph.checkpointer is None

    async def test_checkpointer_is_set_when_provided(self):
        saver = InMemorySaver()
        graph = await build_equity_screening_graph(_CFG, checkpointer=saver)
        assert graph.checkpointer is saver

    async def test_compiles_with_store(self):
        store = InMemoryStore()
        graph = await build_equity_screening_graph(_CFG, store=store)
        assert isinstance(graph, CompiledStateGraph)
        assert graph.store is store

    async def test_default_store_is_none(self):
        graph = await build_equity_screening_graph(_CFG)
        assert graph.store is None

    async def test_per_ticker_worker_is_a_subgraph_node(self):
        """The analysis pipeline runs as a real subgraph node, not via .ainvoke().

        That is the whole point: a subgraph invoked inside a function body is not a
        pregel task, so nothing under it gets a checkpoint namespace.
        """
        graph = await build_equity_screening_graph(_CFG)
        assert "analyze_ticker" in graph.nodes
        nested = graph.get_graph(xray=1)
        assert any("market_regime" in str(n) for n in nested.nodes)


@pytest.mark.unit
@pytest.mark.asyncio
class TestStageAgentsReceiveARunnableConfig:
    """Regression: the stage agents must be built from a real ``RunnableConfig``.

    The deleted ``run_deep_agent_node`` wrapper called
    ``agent_factory(ModelConfiguration.from_runnable_config(config), ...)`` — passing a
    ``ModelConfiguration`` to factories that expect a ``RunnableConfig``. Every stage
    therefore raised ``AttributeError: 'ModelConfiguration' object has no attribute
    'get'`` on its first MCP lookup, and the wrapper's bare ``except Exception``
    swallowed it into ``{"error": "Agent raised an exception"}``. The graph "succeeded"
    while every stage had failed.
    """

    async def test_factories_accept_a_runnable_config(self):
        from muffin_agent.agents.investment import (
            create_company_analysis_agent,
            create_forecasting_agent,
            create_market_regime_agent,
            create_risk_assessment_agent,
            create_sector_analysis_agent,
            create_valuation_agent,
        )

        for factory in (
            create_market_regime_agent,
            create_sector_analysis_agent,
            create_company_analysis_agent,
            create_forecasting_agent,
            create_risk_assessment_agent,
            create_valuation_agent,
        ):
            agent = await factory({"configurable": {}})
            assert agent is not None, factory.__name__
