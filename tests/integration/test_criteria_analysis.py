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
    """Full graph: requested ticker flows through every stage; fan-in is complete."""
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

    # Synthesis produced a final view; no stage left an error payload.
    assert result["synthesis"]["ticker"] == "AAPL"
    assert result["synthesis"]["signal"] == "buy"
    for key in ("classification", "criteria_definition", "valuation_methodology"):
        assert "error" not in result[key]


async def test_classification_stage_renders_ticker_into_system_prompt(config, store):
    """Regression: the requested ticker MUST reach the model (via the system
    prompt now that context is rendered, not a dropped ``input`` key)."""
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
    assert "aapl" in system  # the ticker was rendered into the prompt
    assert "ticker classification agent" in system  # base prompt survived


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
    assert "ticker classification agent" in system.lower()  # base prompt present
    assert "SENTINEL-LESSON" in system  # AND the lesson was appended
