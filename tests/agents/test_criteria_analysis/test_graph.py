"""Smoke tests for the criteria-analysis graph builder."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.memory import InMemoryStore

from muffin_agent.agents.criteria_analysis.graph import (
    _fan_out_criteria,
    build_criteria_analysis_graph,
    make_graph,
)

_CONFIG = {"configurable": {"thread_id": "t"}}

# The five stage-agent factories are heavy (MCP subagents, deep-agent middleware
# stacks). Stub them at the factory seam so the graph builder can be exercised
# without building real agents — mirrors how the trading tests stub the analyst
# builders. Each stub returns a MagicMock the StateGraph treats as a node.
_GRAPH_MOD = "muffin_agent.agents.criteria_analysis.graph"
_FACTORY_NAMES = (
    "create_ticker_classification_agent",
    "create_criteria_definition_agent",
    "create_valuation_methodology_agent",
    "build_criterion_evaluation_worker",
    "create_synthesis_agent",
)
_FACTORY_PATCHES = {
    f"{_GRAPH_MOD}.{name}": AsyncMock(return_value=MagicMock())
    for name in _FACTORY_NAMES
}


def _patch_factories():
    """Context-manager stack patching every stage-agent factory."""
    from contextlib import ExitStack

    stack = ExitStack()
    for target, mock in _FACTORY_PATCHES.items():
        stack.enter_context(patch(target, mock))
    return stack


@pytest.mark.unit
class TestBuildGraph:
    @pytest.mark.asyncio
    async def test_compiles_without_checkpointer_or_store(self):
        with _patch_factories():
            g = await build_criteria_analysis_graph(_CONFIG)
        assert isinstance(g, CompiledStateGraph)

    @pytest.mark.asyncio
    async def test_compiles_with_checkpointer(self):
        with _patch_factories():
            g = await build_criteria_analysis_graph(_CONFIG, checkpointer=MemorySaver())
        assert isinstance(g, CompiledStateGraph)

    @pytest.mark.asyncio
    async def test_compiles_with_store(self):
        with _patch_factories():
            g = await build_criteria_analysis_graph(_CONFIG, store=InMemoryStore())
        assert isinstance(g, CompiledStateGraph)

    @pytest.mark.asyncio
    async def test_make_graph_factory_compiles(self):
        with _patch_factories():
            g = await make_graph(_CONFIG)
        assert isinstance(g, CompiledStateGraph)

    @pytest.mark.asyncio
    async def test_expected_nodes(self):
        with _patch_factories():
            g = await build_criteria_analysis_graph(_CONFIG)
        names = set(g.nodes.keys())
        assert {
            "ticker_classification",
            "lift_classification",
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
        # The reducer seed is no longer sent — the worker subgraph owns it.
        assert "criterion_evaluations" not in payload

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
class TestClassificationRoutingAndLift:
    def test_route_to_lift_when_all_flat_keys_supplied(self):
        from muffin_agent.agents.criteria_analysis.ticker_classification import (
            route_classification_entry,
        )

        state = {
            "ticker": "JPM",
            "sector": "banking",
            "market": "developed",
            "stock_type": "value",
        }
        assert route_classification_entry(state) == "lift_classification"

    def test_route_to_agent_when_missing_flat_keys(self):
        from muffin_agent.agents.criteria_analysis.ticker_classification import (
            route_classification_entry,
        )

        state = {"ticker": "JPM", "sector": "banking", "market": "developed"}
        assert route_classification_entry(state) == "ticker_classification"

    def test_lift_assembles_classification_from_flat_keys(self):
        from muffin_agent.agents.criteria_analysis.ticker_classification import (
            lift_classification_node,
        )

        state = {
            "ticker": "JPM",
            "sector": "banking",
            "market": "developed",
            "stock_type": "value",
        }
        update = lift_classification_node(state)
        assert update["classification"]["confidence"] == 1.0
        assert update["classification"]["sector"] == "banking"
        assert update["classification"]["ticker"] == "JPM"

    def test_lift_flattens_agent_classification(self):
        from muffin_agent.agents.criteria_analysis.ticker_classification import (
            lift_classification_node,
        )

        state = {
            "ticker": "JPM",
            "classification": {
                "ticker": "JPM",
                "sector": "banking",
                "sub_sector": None,
                "market": "developed",
                "stock_type": "value",
            },
        }
        update = lift_classification_node(state)
        assert update["sector"] == "banking"
        assert update["market"] == "developed"
        assert update["stock_type"] == "value"
        assert "classification" not in update  # doesn't re-emit the payload

    def test_lift_raises_when_nothing_to_lift(self):
        from muffin_agent.agents.criteria_analysis.ticker_classification import (
            lift_classification_node,
        )

        with pytest.raises(ValueError, match="no classification"):
            lift_classification_node({"ticker": "JPM"})


@pytest.mark.unit
class TestPackageEvaluationNode:
    def test_augments_and_appends_evaluation(self):
        from muffin_agent.agents.criteria_analysis.criterion_evaluation_node import (
            package_evaluation_node,
        )

        state = {
            "criterion": {"name": "ROE", "weight": 0.4, "source": "skill"},
            "evaluation": {"score": 0.7, "signal": "positive"},
        }
        update = package_evaluation_node(state)  # type: ignore[arg-type]
        evals = update["criterion_evaluations"]
        assert len(evals) == 1
        assert evals[0]["criterion_name"] == "ROE"
        assert evals[0]["weight"] == 0.4
        assert evals[0]["source"] == "skill"
        assert "tool_runs" not in evals[0]  # none captured → not attached

    def test_attaches_per_criterion_tool_runs(self):
        from muffin_agent.agents.criteria_analysis.criterion_evaluation_node import (
            package_evaluation_node,
        )

        state = {
            "criterion": {"name": "ROE", "weight": 0.4, "source": "skill"},
            "evaluation": {"score": 0.7},
            "tool_runs": [{"tool": "equity_fundamentals", "status": "ok"}],
        }
        update = package_evaluation_node(state)  # type: ignore[arg-type]
        evaluation = update["criterion_evaluations"][0]
        assert evaluation["tool_runs"] == [
            {"tool": "equity_fundamentals", "status": "ok"}
        ]


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
