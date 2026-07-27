"""Every graph's execution tree must be reconstructable from checkpoints alone.

This is the invariant the whole observability design rests on, and it is a
property of **how the graph is built**, not of any telemetry:

    a node is natively drillable if and only if it is a compiled agent/subgraph
    added via ``add_node``.

LangGraph gives such a node its own ``checkpoint_ns``, so
``POST /threads/{id}/history`` on that namespace returns its supersteps, its
tasks, and its ``values.messages`` — the transcript with its tool calls. A plain
function node reports ``checkpoint: null``: genuinely leaf-shaped, not a gap.

These tests read ``node.subgraphs``, which is exactly what LangGraph itself
consults when deciding whether to hand a task a child namespace
(``Pregel._prepare_state_snapshot``). No run, no LLM, no network — so a
refactor that quietly turns a compiled agent back into a plain function call
(the "Pattern B" anti-pattern: ``await agent.ainvoke()`` inside a node body,
which is invisible AND unrecoverable) fails here rather than in production.

There used to be a bespoke ``subagent_tree`` capture channel asserted by the
E2E tests. It was deleted: LangGraph already persists all of this. These tests
replace those assertions with the structural property that makes the native
path work.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytestmark = pytest.mark.asyncio


def drillable(graph: Any) -> set[str]:
    """Node names LangGraph will give their own ``checkpoint_ns``."""
    return {
        name for name, node in graph.nodes.items() if getattr(node, "subgraphs", None)
    }


def leaves(graph: Any) -> set[str]:
    """Node names with no child namespace — plain function/LLM nodes."""
    return {
        name
        for name, node in graph.nodes.items()
        if not getattr(node, "subgraphs", None) and not name.startswith("__")
    }


def _no_mcp():
    """Patch MCP so graph construction never touches the network."""
    client = MagicMock()
    client.get_tools = AsyncMock(return_value=[])
    return patch(
        "muffin_agent.agents.data_collection.utils.MultiServerMCPClient",
        MagicMock(return_value=client),
    )


async def test_trading_decision_analysts_and_debates_are_drillable(config):
    """The four analysts and both debate conferences expose namespaces.

    Verified against production thread ``019f81a0``: ``market_analyst:<uuid>``
    returns 13 messages (1 human / 2 ai / 10 tool) with 10 real tool calls.
    """
    from muffin_agent.agents.trading_decision.graph import (
        build_trading_decision_graph,
    )

    with _no_mcp():
        graph = await build_trading_decision_graph(config)

    assert {
        "market_analyst",
        "fundamentals_analyst",
        "news_analyst",
        "social_analyst",
        "investment_debate",
        "risk_debate",
    } <= drillable(graph)

    # Single structured LLM calls. Their output lands in the parent's values, so
    # showing name + status + result is honest — there is no sub-structure to
    # drill into. Listed explicitly so promoting one to an agent is a deliberate
    # change to this test, not a silent drift.
    assert {"investment_judge", "trader", "portfolio_manager"} <= leaves(graph)


async def test_council_personas_are_drillable(config, store):
    """Every persona is a compiled subgraph, so its own work is inspectable.

    This is what makes "click a persona → see how its data-collection sub-agent
    worked" possible without any capture channel.
    """
    from muffin_agent.agents.personas_council.council_graph import (
        PERSONA_BUILDERS,
        build_council_graph,
    )

    with _no_mcp():
        graph = await build_council_graph(config, store=store)

    inspectable = drillable(graph)
    missing = {slug for slug, _ in PERSONA_BUILDERS} - inspectable
    assert not missing, f"personas invisible to checkpoint history: {missing}"


async def test_criteria_stages_and_workers_are_drillable(config, store):
    """Each stage agent and the Send fan-out worker expose namespaces."""
    from muffin_agent.agents.criteria_analysis.graph import (
        build_criteria_analysis_graph,
    )

    with _no_mcp():
        graph = await build_criteria_analysis_graph(config, store=store)

    assert {
        "ticker_classification",
        "criteria_definition",
        "valuation_methodology",
        "criterion_evaluation",
        "synthesis",
    } <= drillable(graph)


async def test_research_stages_are_drillable(config, store):
    """The research stages are compiled agent nodes, not in-body ``ainvoke``.

    Regression guard for the Pattern B migration: these three used to call
    ``agent.ainvoke()`` inside a plain function node, which made them black
    boxes — the inner agent was not a pregel task, so nothing was checkpointed
    and the work was unrecoverable after the fact.
    """
    from muffin_agent.agents.research.graph import build_research_graph

    with _no_mcp():
        graph = await build_research_graph(config, store=store)

    inspectable = drillable(graph)
    assert "classifier" in inspectable
    assert "writer" in inspectable
    assert any("research" in name for name in inspectable), (
        f"no researcher node is drillable: {sorted(inspectable)}"
    )


async def test_no_graph_smuggles_observability_through_state(config, store):
    """The capture channels are gone and must not come back.

    ``tool_runs`` / ``subagent_tree`` / ``subagent_runs`` were reducer channels
    that every ``output_schema`` boundary had to re-declare with the exact same
    reducer or a whole subtree vanished silently. Observability belongs in
    checkpoints; a channel exists because a *node* needs it for logic.
    """
    from muffin_agent.agents.criteria_analysis.graph import (
        build_criteria_analysis_graph,
    )
    from muffin_agent.agents.personas_council.council_graph import build_council_graph
    from muffin_agent.agents.trading_decision.graph import (
        build_trading_decision_graph,
    )

    banned = {"tool_runs", "subagent_tree", "subagent_runs"}
    with _no_mcp():
        graphs = {
            "trading_decision": await build_trading_decision_graph(config),
            "council": await build_council_graph(config, store=store),
            "criteria_analysis": await build_criteria_analysis_graph(
                config, store=store
            ),
        }

    for name, graph in graphs.items():
        channels = set(graph.output_schema.model_json_schema().get("properties", {}))
        assert not (channels & banned), (
            f"{name} exposes observability-as-state again: {channels & banned}"
        )
