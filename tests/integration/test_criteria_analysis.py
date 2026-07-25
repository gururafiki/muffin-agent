"""E2E integration test — the criteria-driven analysis orchestrator graph.

The deployable ``"criteria_analysis"`` graph (langgraph.json) is now built from
compiled agents added directly as graph nodes (the trading-analyst / council
pattern). This test proves the whole pipeline wires up and — critically — that
the requested ticker actually reaches every stage (the bug this refactor fixed:
the old nodes invoked agents with ``{"input": ...}``, which LangChain silently
ignores, so the model saw zero messages and hallucinated a different ticker).

Parallel Stage 2/3 + the Send fan-out over criteria interleave model calls, so
the graph is driven with the **schema-routed** model (``patch_llm_by_schema``) —
it answers by the bound response schema, not call order.

Two focused node tests lock the regressions directly:

* ``test_classification_stage_renders_ticker_into_system_prompt`` — the task
  context (ticker) must appear in the rendered system prompt (would have caught
  the empty-messages bug).
* ``test_lessons_block_composes_not_replaces_system_prompt`` — a seeded lesson
  must be APPENDED to the base prompt, not replace it (the content-blocks wipe).
"""

from __future__ import annotations

from typing import Any

import pytest

from muffin_agent.agents.criteria_analysis.graph import build_criteria_analysis_graph
from muffin_agent.agents.criteria_analysis.schemas import (
    CriteriaAnalysisSynthesis,
    SynthesisNodeOutput,
    TickerClassificationNodeOutput,
    TickerClassificationOutput,
    ValuationMethodologyNodeOutput,
    ValuationMethodologyOutput,
)
from muffin_agent.agents.criteria_definition import (
    CriteriaDefinitionNodeOutput,
    CriteriaDefinitionOutput,
    ValuationCriterion,
)
from muffin_agent.agents.criterion_evaluation import (
    CriterionEvaluationNodeOutput,
    CriterionEvaluationOutput,
)

from ._harness import (
    patch_llm,
    patch_llm_by_schema,
    patch_mcp,
    patch_sandbox,
    tool_turn,
)

pytestmark = pytest.mark.asyncio


def _criterion(name: str, weight: float) -> ValuationCriterion:
    return ValuationCriterion(
        name=name,
        target_range="0.8-2.0x",
        weight=weight,
        assessment_guidance="Strong below range, weak above.",
        data_requirements=["equity-fundamentals"],
    )


def _schema_responses() -> dict[Any, Any]:
    """Full, valid structured responses keyed by each stage's wrapper schema."""
    classification = TickerClassificationOutput(
        ticker="AAPL",
        sector="software-saas",
        sub_sector=None,
        market="developed",
        stock_type="growth",
        rationale="Test classification.",
        confidence=0.9,
    )
    criteria_definition = CriteriaDefinitionOutput(
        ticker="AAPL",
        sector="Technology - Software/SaaS",
        market_type="developed",
        stock_type="growth",
        classification_rationale="Test.",
        primary_valuation_method="P/E + EV/EBITDA",
        criteria=[_criterion("Revenue Growth", 0.5), _criterion("Gross Margin", 0.5)],
        screening_questions=["Is growth durable?"],
        valuation_errors_to_avoid=["Ignoring dilution"],
        confidence=0.8,
    )
    methodology = ValuationMethodologyOutput(
        ticker="AAPL",
        methodology_summary="DCF + peer multiples.",
        additional_criteria=[_criterion("Services Mix Shift", 0.2)],
    )
    evaluation = CriterionEvaluationOutput(
        criterion_name="(scored)",
        score=0.7,
        confidence=0.8,
        signal="positive",
        sub_criteria=[],
        evidence_summary=["3Y revenue CAGR 12% vs sector 8%."],
        reasoning="Growth is above the sector median.",
        counterargument="Deceleration risk.",
    )
    synthesis = CriteriaAnalysisSynthesis(
        ticker="AAPL",
        composite_score=0.7,
        signal="buy",
        weighted_breakdown=[],
        key_positives=["Durable growth."],
        key_negatives=["Valuation rich."],
        confidence=0.75,
        thesis_paragraph="A buy on durable growth.",
    )
    return {
        "TickerClassificationNodeOutput": TickerClassificationNodeOutput(
            classification=classification
        ),
        "CriteriaDefinitionNodeOutput": CriteriaDefinitionNodeOutput(
            criteria_definition=criteria_definition
        ),
        "ValuationMethodologyNodeOutput": ValuationMethodologyNodeOutput(
            valuation_methodology=methodology
        ),
        "CriterionEvaluationNodeOutput": CriterionEvaluationNodeOutput(
            evaluation=evaluation
        ),
        "SynthesisNodeOutput": SynthesisNodeOutput(synthesis=synthesis),
    }


async def test_criteria_analysis_runs_end_to_end(config, store):
    """Full graph: requested ticker flows through every stage; fan-in is complete.

    Capture is always on, exercising the ``tool_runs`` plumbing through every
    node (incl. the worker subgraph's restricted ``output_schema``) — it must
    not break the run. (The schema-routed model ends each agent on its first
    call, so no data-collection tool calls fire here; capture semantics are
    covered by ``tests/middlewares/test_agent_capture_records.py``.)
    """
    responses = _schema_responses()
    with patch_mcp(scenario="aapl"), patch_sandbox(), patch_llm_by_schema(responses):
        graph = await build_criteria_analysis_graph(config, store=store)
        result = await graph.ainvoke(
            {"ticker": "AAPL", "query": "Long-term hold?"}, config=config
        )

    # The requested ticker reached classification — NOT a hallucinated one.
    assert result["classification"]["ticker"] == "AAPL"
    # Flat keys were lifted for the skill filter.
    assert result["sector"] == "software-saas"
    assert result["market"] == "developed"

    # Two skill criteria + one web criterion (distinct names) → 3 merged, 3 evals.
    assert len(result["merged_criteria"]) == 3
    evals = result["criterion_evaluations"]
    assert len(evals) == 3
    names = {e["criterion_name"] for e in evals}
    assert names == {"Revenue Growth", "Gross Margin", "Services Mix Shift"}
    for e in evals:
        assert "weight" in e and "source" in e
        assert e["source"] in {"skill", "web"}
        # The schema-routed model made zero tool calls, so the package node's
        # data-source reconciliation flags every evaluation as prior-knowledge.
        assert e["data_collected"] is False
        assert e["data_sources"] == []

    # Synthesis produced a final view; no stage left an error payload.
    assert result["synthesis"]["ticker"] == "AAPL"
    assert result["synthesis"]["signal"] == "buy"
    for key in ("classification", "criteria_definition", "valuation_methodology"):
        assert "error" not in result[key]

    # subagent_tree — top level: one node per stage agent (classification,
    # criteria_definition, valuation_methodology, synthesis), each added
    # directly to the orchestrator graph, so each roots at depth 1.
    tree = result.get("subagent_tree") or {}
    assert tree, "subagent_tree should be populated"
    assert len(tree) == 4
    names = {n["name"] for n in tree.values()}
    assert names == {
        "ticker_classification",
        "criteria_definition",
        "valuation_methodology",
        "criteria_analysis_synthesis",
    }
    parents = {n["parent_id"] for n in tree.values()}
    assert parents == {"__root__"}  # depth-1 nodes rooted correctly
    assert all("|" not in node_id for node_id in tree)  # not nested — no parent agent

    # subagent_tree — per criterion: the criterion_evaluation deep agent runs
    # inside the Send-fan-out worker subgraph (evaluate -> package), so its
    # tree node id is nested (worker ns + its own node), and the worker's
    # `package` node moves it onto the evaluation dict rather than the
    # top-level channel (kept scoped per-criterion, see criterion_evaluation
    # _node.py / state.py). This is where real nesting (id containing "|")
    # shows up for this graph.
    for e in evals:
        per_criterion_tree = e.get("subagent_tree") or {}
        assert per_criterion_tree, "each criterion's subagent_tree should be populated"
        [criterion_node] = list(per_criterion_tree.values())
        assert criterion_node["name"] == "criterion_evaluation"
        assert "|" in criterion_node["id"]  # nested under the worker's `evaluate` node
        assert criterion_node["parent_id"] != "__root__"


async def test_classification_stage_renders_ticker_into_human_message(config, store):
    """Regression: the requested ticker MUST reach the model — now via the FIRST
    human message (input template), never baked into the system prompt (which
    would make Ollama Cloud 500 on the system-only call)."""
    from muffin_agent.agents.criteria_analysis.ticker_classification import (
        create_ticker_classification_agent,
    )

    with (
        patch_mcp(scenario="aapl"),
        patch_sandbox(),
        patch_llm(
            tool_turn(
                "TickerClassificationNodeOutput",
                {
                    "classification": {
                        "ticker": "AAPL",
                        "sector": "software-saas",
                        "sub_sector": None,
                        "market": "developed",
                        "stock_type": "growth",
                        "rationale": "t",
                        "confidence": 0.9,
                        "data_sources": [],
                        "limitations": [],
                    }
                },
            )
        ) as cursor,
    ):
        agent = await create_ticker_classification_agent(config, store=store)
        await agent.ainvoke({"ticker": "AAPL", "query": "hold?"}, config=config)

    system = cursor.last_system_prompt().lower()
    human = cursor.last_human_prompt().lower()
    assert "aapl" in human  # the ticker was rendered into the first human message
    assert "ticker classification agent" in human  # the task template is the user turn
    assert "aapl" not in system  # user input is NOT baked into the system prompt


async def test_lessons_block_composes_not_replaces_system_prompt(config, store):
    """Regression: a seeded lesson is APPENDED to the base prompt, never
    replaces it (the content-blocks wipe that mis-classified NVDA as AAPL)."""
    from muffin_agent.agents.criteria_analysis.ticker_classification import (
        create_ticker_classification_agent,
    )

    await store.aput(
        ("tool_lessons", "write_todos"),
        "seed",
        {
            "tool_name": "write_todos",
            "lesson": "SENTINEL-LESSON keep the base prompt",
            "created_at": "2026-01-01T00:00:00+00:00",
        },
    )

    with (
        patch_mcp(scenario="aapl"),
        patch_sandbox(),
        patch_llm(
            tool_turn(
                "TickerClassificationNodeOutput",
                {
                    "classification": {
                        "ticker": "AAPL",
                        "sector": "software-saas",
                        "sub_sector": None,
                        "market": "developed",
                        "stock_type": "growth",
                        "rationale": "t",
                        "confidence": 0.9,
                        "data_sources": [],
                        "limitations": [],
                    }
                },
            )
        ) as cursor,
    ):
        agent = await create_ticker_classification_agent(config, store=store)
        await agent.ainvoke({"ticker": "AAPL", "query": "hold?"}, config=config)

    system = cursor.last_system_prompt()
    human = cursor.last_human_prompt().lower()
    assert "SENTINEL-LESSON" in system  # the lesson was appended to the system prompt
    assert "ticker classification agent" in human  # the task template is the user turn
    # The framework base (partials / deepagents base) survived the lesson compose
    # — it was NOT wiped by a naive str-only append over content-blocks content.
    assert "/scratch/" in system.lower()
