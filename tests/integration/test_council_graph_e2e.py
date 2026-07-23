"""E2E integration test — the full persona council graph.

The deployable ``"council"`` graph (langgraph.json) fans out to all **13 persona
subgraphs in parallel**, each a real ``collect_data → compute_evidence →
render_verdict`` pipeline, then aggregates into the council judge. This is the
test that proves the compiled-subagent composition fix at scale:

* Each persona is added to the council via ``input_schema=PersonaInput`` (the fix)
  and maps ``CouncilState → persona`` across differing schemas.
* The 13 personas run concurrently, so we drive them with the **schema-routed**
  model (``patch_llm_by_schema``) — it answers by the bound response schema, not
  call order, so parallel interleaving is irrelevant.
* ``compute_evidence`` runs for real inside each persona (never mocked); the
  scripted verdict reuses each persona's real evidence type.

Before the fix this raised ``ValidationError`` at the first persona node.
"""

from __future__ import annotations

import importlib
from typing import Any

import pytest
from pydantic import BaseModel

from muffin_agent.agents.personas_council.council_graph import (
    PERSONA_BUILDERS,
    build_council_graph,
)
from muffin_agent.agents.personas_council.judge import CouncilSynthesisOutput
from muffin_agent.agents.personas_council.schemas import AnalystSignal, InvestmentSignal

from ._harness import patch_llm_by_schema, patch_mcp, patch_sandbox

pytestmark = pytest.mark.asyncio


def _persona_schemas(slug: str, builder: Any) -> tuple[str, type[AnalystSignal], Any]:
    """Find a persona's RawData schema name, Signal class, and compute fn."""
    module = importlib.import_module(builder.__module__)
    raw_cls = next(
        v
        for v in vars(module).values()
        if isinstance(v, type)
        and issubclass(v, BaseModel)
        and v.__name__.endswith("RawData")
    )
    signal_cls = next(
        v
        for v in vars(module).values()
        if isinstance(v, type)
        and issubclass(v, AnalystSignal)
        and v is not AnalystSignal
    )
    return raw_cls.__name__, signal_cls, module.compute_evidence_node


def _build_schema_responses() -> dict[Any, Any]:
    """Map every persona's RawData (→ empty defaults) and Signal (→ instance).

    The Signal reuses each persona's REAL ``compute_evidence_node`` on empty
    state (every persona has a graceful all-zero fallback), so the verdict the
    scripted model returns is a genuinely-valid persona signal.
    """
    responses: dict[Any, Any] = {}
    for slug, builder in PERSONA_BUILDERS:
        raw_name, signal_cls, compute = _persona_schemas(slug, builder)
        responses[raw_name] = {}  # ReAct response_format turn → RawData() defaults
        evidence = compute({})["evidence"]  # real deterministic compute, empty input
        responses[signal_cls] = signal_cls(
            signal="hold",
            confidence=0.5,
            reasoning=f"{slug} test verdict.",
            evidence=evidence,
        )
    # The judge's direct structured call.
    responses[CouncilSynthesisOutput] = CouncilSynthesisOutput(
        ticker="AAPL",
        consensus_rating="hold",
        weighted_confidence=0.5,
        bull_case_synthesis="bull",
        bear_case_synthesis="bear",
        dissent_summary="none",
        reasoning="test synthesis",
    )
    return responses


async def test_council_runs_all_13_personas_to_judge(config, store):
    """build_council_graph + ainvoke: 13 parallel personas aggregate into a verdict."""
    responses = _build_schema_responses()
    with patch_mcp(scenario="aapl"), patch_sandbox(), patch_llm_by_schema(responses):
        graph = await build_council_graph(config, store=store)
        result = await graph.ainvoke(
            {"ticker": "AAPL", "as_of_date": "2026-06-09", "query": None},
            config=config,
        )

    # All 13 personas ran end-to-end and emitted a signal (operator.add aggregate).
    signals = result["persona_signals"]
    assert len(signals) == len(PERSONA_BUILDERS) == 13
    agent_ids = {s["agent_id"] for s in signals}
    assert len(agent_ids) == 13  # every persona is distinct — none crashed/dropped

    # The judge synthesised the ensemble.
    synthesis = result["council_synthesis"]
    assert synthesis["consensus_rating"] in InvestmentSignal.__args__
    assert synthesis["ticker"] == "AAPL"

    # subagent_tree: one node per persona's collect_data ReAct agent (the only
    # capturing node in each 3-node persona subgraph — compute_evidence/
    # render_verdict are plain/LLM-direct, not agents). Each id is
    # "<slug>:<task_id>|collect_data:<task_id>" — two real ns segments deep
    # (persona subgraph, then its own collect_data agent), which is why the
    # `|` proves real nesting even though the persona-subgraph level itself
    # never gets a tree entry (nothing captures there; the consumer
    # reconstructs it from the id prefix, per tree.py's docstring). Root-level
    # ("__root__") rooting is covered by test_criteria_analysis.py instead —
    # every capturing node here sits one level deeper than that.
    tree = result.get("subagent_tree") or {}
    assert tree, "subagent_tree should be populated"
    assert len(tree) == 13  # one per persona, none dropped/collided
    names = {n["name"] for n in tree.values()}
    assert {"warren_buffett_data_collection", "mohnish_pabrai_data_collection"} <= names
    assert all("_data_collection" in n for n in names)
    assert all("|" in node_id for node_id in tree)  # nested under the persona subgraph
    parents = {n["parent_id"] for n in tree.values()}
    assert len(parents) == 13  # each persona's collect_data has its own distinct parent
    assert "__root__" not in parents  # by design: see comment above

    # No persona's internal scratch (the now-``output=False`` collect_data fields,
    # ``evidence``, ``structured_response``, ``messages``) leaks into the council
    # state — only CouncilState channels survive (the persona graphs' explicit
    # ``output_schema=<Persona>Output`` filters them at the council boundary).
    # `tool_runs` / `subagent_tree` are legitimate CouncilState channels: each
    # persona's `<Persona>Output` surfaces its collect_data agent's records
    # (tool_runs empty here — the schema-routed harness single-shots the
    # RawData response with no MCP tool calls, so no tool records are
    # captured — but subagent_tree is captured unconditionally, so it's
    # always present) and the council accumulates them.
    allowed = {
        "ticker",
        "query",
        "as_of_date",
        "persona_signals",
        "council_synthesis",
        "tool_runs",
        "subagent_tree",
    }
    assert set(result.keys()) <= allowed, (
        f"persona internals leaked into council state: {set(result.keys()) - allowed}"
    )
