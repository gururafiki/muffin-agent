"""Smoke tests for the criteria-analysis graph builder."""

from unittest.mock import MagicMock

import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.memory import InMemoryStore

from muffin_agent.agents.criteria_analysis.graph import (
    _fan_out_criteria,
    build_criteria_analysis_graph,
    graph,
)


@pytest.mark.unit
class TestBuildGraph:
    def test_module_level_graph_is_compiled(self):
        assert isinstance(graph, CompiledStateGraph)

    def test_compiles_without_checkpointer_or_store(self):
        g = build_criteria_analysis_graph()
        assert isinstance(g, CompiledStateGraph)

    def test_compiles_with_checkpointer(self):
        cp = MemorySaver()
        g = build_criteria_analysis_graph(checkpointer=cp)
        assert isinstance(g, CompiledStateGraph)

    def test_compiles_with_store(self):
        store = InMemoryStore()
        g = build_criteria_analysis_graph(store=store)
        assert isinstance(g, CompiledStateGraph)

    def test_expected_nodes(self):
        g = build_criteria_analysis_graph()
        names = set(g.nodes.keys())
        assert {
            "ticker_classification",
            "criteria_definition",
            "valuation_methodology",
            "merge_criteria",
            "criterion_evaluation",
            "synthesis",
        } <= names


@pytest.mark.unit
class TestFanOut:
    def test_one_send_per_merged_criterion(self):
        state: dict = {
            "ticker": "JPM",
            "query": "value bank thesis",
            "classification": {"sector": "banking"},
            "merged_criteria": [
                {"name": "ROE", "weight": 0.4, "source": "skill"},
                {"name": "P/B", "weight": 0.3, "source": "skill"},
                {"name": "Activist Pressure", "weight": 0.3, "source": "web"},
            ],
        }
        sends = _fan_out_criteria(state)  # type: ignore[arg-type]
        assert len(sends) == 3
        assert all(s.node == "criterion_evaluation" for s in sends)

    def test_send_payload_carries_context(self):
        state: dict = {
            "ticker": "AAPL",
            "query": "growth thesis",
            "classification": {"sector": "consumer-discretionary"},
            "merged_criteria": [
                {"name": "Services Mix", "weight": 1.0, "source": "web"},
            ],
        }
        sends = _fan_out_criteria(state)  # type: ignore[arg-type]
        payload = sends[0].arg
        assert payload["ticker"] == "AAPL"
        assert payload["query"] == "growth thesis"
        assert payload["classification"]["sector"] == "consumer-discretionary"
        assert payload["criterion"]["name"] == "Services Mix"
        assert payload["criterion_evaluations"] == []

    def test_no_merged_criteria_sends_nothing(self):
        state: dict = {
            "ticker": "X",
            "query": "q",
            "classification": {},
            "merged_criteria": [],
        }
        sends = _fan_out_criteria(state)  # type: ignore[arg-type]
        assert sends == []


@pytest.mark.unit
class TestTickerClassificationShortCircuit:
    def test_short_circuit_when_all_flat_keys_supplied(self):
        from muffin_agent.agents.criteria_analysis.ticker_classification import (
            _shortcircuit,
        )

        state = {
            "ticker": "JPM",
            "sector": "banking",
            "market": "developed",
            "stock_type": "value",
        }
        update = _shortcircuit(state)
        assert update is not None
        assert update["sector"] == "banking"
        assert update["classification"]["confidence"] == 1.0

    def test_no_short_circuit_when_missing_flat_keys(self):
        from muffin_agent.agents.criteria_analysis.ticker_classification import (
            _shortcircuit,
        )

        # Missing stock_type
        state = {"ticker": "JPM", "sector": "banking", "market": "developed"}
        assert _shortcircuit(state) is None

    def test_no_short_circuit_when_empty(self):
        from muffin_agent.agents.criteria_analysis.ticker_classification import (
            _shortcircuit,
        )

        assert _shortcircuit({"ticker": "JPM"}) is None


@pytest.mark.unit
class TestMergeCriteriaNode:
    @pytest.mark.asyncio
    async def test_merge_node_reads_upstream_state(self):
        from muffin_agent.agents.criteria_analysis.merge_criteria import (
            merge_criteria_node,
        )

        state: dict = {
            "criteria_definition": {
                "criteria": [
                    {
                        "name": "ROE",
                        "target_range": "10-15%",
                        "weight": 0.5,
                        "assessment_guidance": "g",
                        "data_requirements": ["equity-fundamentals"],
                    }
                ],
            },
            "valuation_methodology": {
                "additional_criteria": [
                    {
                        "name": "Activist Pressure",
                        "target_range": "no",
                        "weight": 0.5,
                        "assessment_guidance": "g",
                        "data_requirements": ["news"],
                    }
                ],
            },
        }
        update = await merge_criteria_node(state, MagicMock())
        merged = update["merged_criteria"]
        assert len(merged) == 2
        assert sum(c["weight"] for c in merged) == pytest.approx(1.0, abs=1e-9)
