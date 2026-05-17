"""End-to-end tests for the investment debate graph.

Patches the three agent factories with deterministic fakes so the graph
runs to completion without an LLM call. Verifies:

* The graph compiles successfully.
* A default-rounds run produces a full Bull→Bear→Bull→Bear→Judge sequence
  and writes ``investment_judge`` to the final state.
* The per-run ``max_investment_debate_rounds`` override is honoured.
* The judge's structured output is preserved in the final state.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.messages import AIMessage

from muffin_agent.agents.trading_decision import (
    AnalysisContext,
    InvestmentJudgeOutput,
    build_investment_debate_graph,
)
from muffin_agent.agents.trading_decision.conditional_logic import (
    BEAR_TAG,
    BULL_TAG,
)


def _judge_payload(**overrides) -> dict:
    base = {
        "signal": "buy",
        "conviction": 0.75,
        "summary": "Net bull.",
        "bull_case": "Bull held.",
        "bear_case": "Bear had real points.",
        "key_catalysts": ["catalyst A"],
        "key_risks": ["risk A"],
        "monitoring_checklist": ["metric A"],
        "winning_side": "bull",
        "reasoning": "Bull addressed bear points.",
    }
    base.update(overrides)
    return base


def _judge_result(payload: dict) -> dict:
    return {
        "messages": [AIMessage(content="structured ok")],
        "structured_response": InvestmentJudgeOutput.model_validate(payload),
    }


def _fake_factories(
    *,
    bull_reply: str = "Bull argument.",
    bear_reply: str = "Bear argument.",
    judge_payload: dict | None = None,
):
    """Return three AsyncMocks for the bull/bear/judge factories.

    Each factory mock returns an agent whose ``ainvoke`` is also a mock —
    bull and bear agents return a plain ``AIMessage`` result, judge returns
    a structured response.
    """
    bull_agent = AsyncMock()
    bull_agent.ainvoke.return_value = {"messages": [AIMessage(content=bull_reply)]}
    bear_agent = AsyncMock()
    bear_agent.ainvoke.return_value = {"messages": [AIMessage(content=bear_reply)]}
    judge_agent = AsyncMock()
    judge_agent.ainvoke.return_value = _judge_result(judge_payload or _judge_payload())

    return (
        AsyncMock(return_value=bull_agent),
        AsyncMock(return_value=bear_agent),
        AsyncMock(return_value=judge_agent),
        bull_agent,
        bear_agent,
        judge_agent,
    )


def _patch_factories(bull_factory, bear_factory, judge_factory):
    return [
        patch(
            "muffin_agent.agents.trading_decision.nodes.create_bull_researcher_agent",
            bull_factory,
        ),
        patch(
            "muffin_agent.agents.trading_decision.nodes.create_bear_researcher_agent",
            bear_factory,
        ),
        patch(
            "muffin_agent.agents.trading_decision.nodes.create_investment_judge_agent",
            judge_factory,
        ),
    ]


@pytest.mark.unit
class TestBuildInvestmentDebateGraph:
    def test_graph_compiles(self):
        graph = build_investment_debate_graph()
        assert graph is not None
        # Compiled graph exposes the underlying state graph nodes
        node_names = set(graph.get_graph().nodes.keys())
        assert {"bull_researcher", "bear_researcher", "investment_judge"} <= node_names


@pytest.mark.unit
@pytest.mark.asyncio
class TestInvestmentDebateExecution:
    async def test_default_rounds_runs_full_sequence(self):
        (
            bull_factory,
            bear_factory,
            judge_factory,
            bull_agent,
            bear_agent,
            judge_agent,
        ) = _fake_factories(
            bull_reply="Bull argument.",
            bear_reply="Bear argument.",
        )

        ctx_managers = _patch_factories(bull_factory, bear_factory, judge_factory)
        for cm in ctx_managers:
            cm.start()
        try:
            graph = build_investment_debate_graph()
            ctx = AnalysisContext.from_narrative("AAPL", "AAPL notes.")
            final_state: dict[str, Any] = await graph.ainvoke(
                {"analysis_context": ctx.model_dump()},
                config={"configurable": {"thread_id": "test-default"}},
            )
        finally:
            for cm in ctx_managers:
                cm.stop()

        # Default = 2 rounds: Bull, Bear, Bull, Bear = 4 turns.
        debate = final_state["investment_debate"]
        assert debate["count"] == 4
        # Each side should have been called twice.
        assert bull_agent.ainvoke.await_count == 2
        assert bear_agent.ainvoke.await_count == 2
        # Judge runs exactly once.
        assert judge_agent.ainvoke.await_count == 1

        # History interleaves Bull / Bear in order.
        history = debate["history"]
        assert history.count(BULL_TAG) == 2
        assert history.count(BEAR_TAG) == 2
        assert history.index(BULL_TAG) < history.index(BEAR_TAG)

        # Judge output captured.
        judge = final_state["investment_judge"]
        assert judge["signal"] == "buy"
        assert judge["winning_side"] == "bull"
        assert judge["conviction"] == 0.75

    async def test_single_round_override_runs_2_turns(self):
        (
            bull_factory,
            bear_factory,
            judge_factory,
            bull_agent,
            bear_agent,
            judge_agent,
        ) = _fake_factories()

        ctx_managers = _patch_factories(bull_factory, bear_factory, judge_factory)
        for cm in ctx_managers:
            cm.start()
        try:
            graph = build_investment_debate_graph()
            ctx = AnalysisContext.from_narrative("AAPL", "Notes.")
            final_state = await graph.ainvoke(
                {"analysis_context": ctx.model_dump()},
                config={
                    "configurable": {
                        "thread_id": "test-1round",
                        "max_investment_debate_rounds": 1,
                    }
                },
            )
        finally:
            for cm in ctx_managers:
                cm.stop()

        debate = final_state["investment_debate"]
        # 1 round = Bull, Bear = 2 turns.
        assert debate["count"] == 2
        assert bull_agent.ainvoke.await_count == 1
        assert bear_agent.ainvoke.await_count == 1
        assert judge_agent.ainvoke.await_count == 1

    async def test_three_rounds_override_runs_6_turns(self):
        (
            bull_factory,
            bear_factory,
            judge_factory,
            bull_agent,
            bear_agent,
            judge_agent,
        ) = _fake_factories()

        ctx_managers = _patch_factories(bull_factory, bear_factory, judge_factory)
        for cm in ctx_managers:
            cm.start()
        try:
            graph = build_investment_debate_graph()
            ctx = AnalysisContext.from_narrative("AAPL", "Notes.")
            final_state = await graph.ainvoke(
                {"analysis_context": ctx.model_dump()},
                config={
                    "configurable": {
                        "thread_id": "test-3rounds",
                        "max_investment_debate_rounds": 3,
                    }
                },
            )
        finally:
            for cm in ctx_managers:
                cm.stop()

        debate = final_state["investment_debate"]
        assert debate["count"] == 6
        assert bull_agent.ainvoke.await_count == 3
        assert bear_agent.ainvoke.await_count == 3

    async def test_strong_sell_signal_propagates(self):
        bull_factory, bear_factory, judge_factory, *_ = _fake_factories(
            judge_payload=_judge_payload(
                signal="strong_sell",
                winning_side="bear",
                conviction=0.85,
            )
        )

        ctx_managers = _patch_factories(bull_factory, bear_factory, judge_factory)
        for cm in ctx_managers:
            cm.start()
        try:
            graph = build_investment_debate_graph()
            ctx = AnalysisContext.from_narrative("TSLA", "TSLA notes.")
            final_state = await graph.ainvoke(
                {"analysis_context": ctx.model_dump()},
                config={"configurable": {"thread_id": "test-strong-sell"}},
            )
        finally:
            for cm in ctx_managers:
                cm.stop()

        judge = final_state["investment_judge"]
        assert judge["signal"] == "strong_sell"
        assert judge["winning_side"] == "bear"
        assert judge["conviction"] == 0.85
