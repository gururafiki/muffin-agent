"""End-to-end tests for the three trading_decision graph builders."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest
from langgraph.store.memory import InMemoryStore

from muffin_agent.agents.trading_decision import (
    Outcome,
    build_investment_debate_graph,
    build_investment_thesis_graph,
    build_trading_decision_graph,
)
from muffin_agent.agents.trading_decision.schemas import (
    InvestmentJudgeOutput,
    PortfolioDecisionOutput,
    TraderOutput,
)

from .conftest import FakeLLM, ai


def _judge_output() -> InvestmentJudgeOutput:
    return InvestmentJudgeOutput.model_validate(
        {
            "signal": "buy",
            "conviction": 0.7,
            "summary": "Net bull.",
            "bull_case": "Bull held.",
            "bear_case": "Bear had points.",
            "key_catalysts": ["catalyst"],
            "key_risks": ["risk"],
            "monitoring_checklist": ["metric"],
            "winning_side": "bull",
            "reasoning": "Reasoning.",
        }
    )


def _trader_output() -> TraderOutput:
    return TraderOutput.model_validate(
        {
            "action": "buy",
            "reasoning": "Starter long.",
            "entry_price": 195.0,
            "stop_loss": 180.0,
            "take_profit": 225.0,
            "position_sizing": "2% NAV",
            "time_horizon": "3-6 months",
        }
    )


def _pm_output(**overrides) -> PortfolioDecisionOutput:
    base = {
        "rating": "buy",
        "executive_summary": "Buy AAPL.",
        "investment_thesis": "Bull held.",
        "price_target": 220.0,
        "stop_loss": 180.0,
        "time_horizon": "3-6 months",
        "position_sizing": "2% NAV starter",
        "key_risks_remaining": ["China"],
        "confidence": 0.7,
    }
    base.update(overrides)
    return PortfolioDecisionOutput.model_validate(base)


def _make_role_factory(role_responses: dict[str, Any]):
    """Build a fake ``ModelConfiguration.from_runnable_config`` whose
    ``get_llm_for_role`` returns a FakeLLM with a per-role response.

    The Mock is built so each invocation returns a FRESH FakeLLM (so
    invocations accumulate per call but the next call gets the response
    we want).
    """
    # Track which roles have been called to give the right response.
    # Since all roles share role="reasoner", we step through a queue.
    response_queue: list[Any] = role_responses["sequence"]
    counter = {"i": 0}

    class FakeConfig:
        def get_llm_for_role(self, role: str):  # noqa: ARG002
            i = counter["i"]
            response = response_queue[i] if i < len(response_queue) else ai("default")
            counter["i"] += 1
            return [FakeLLM(response)]

    def factory(_config):
        return FakeConfig()

    return factory


def _patch_model_config(factory):
    """Patch ``ModelConfiguration.from_runnable_config`` at the source.

    Every per-node module imports the same ``ModelConfiguration`` class, so
    one patch at the source class covers them all without the
    multi-patch leak that would happen if we patched per-module.
    """
    from muffin_agent.model_config import ModelConfiguration

    return patch.object(
        ModelConfiguration, "from_runnable_config", side_effect=factory
    )


# ── Graph compilation tests ──────────────────────────────────────────────────


@pytest.mark.unit
class TestGraphCompilation:
    def test_investment_debate_graph_compiles(self):
        g = build_investment_debate_graph()
        nodes = set(g.get_graph().nodes.keys())
        assert {"bull_researcher", "bear_researcher", "investment_judge"} <= nodes

    def test_investment_thesis_graph_compiles(self):
        g = build_investment_thesis_graph()
        nodes = set(g.get_graph().nodes.keys())
        assert {
            "bull_researcher",
            "bear_researcher",
            "investment_judge",
            "trader",
        } <= nodes

    def test_trading_decision_graph_compiles(self):
        g = build_trading_decision_graph()
        nodes = set(g.get_graph().nodes.keys())
        assert {
            "reflector_resolve",
            "bull_researcher",
            "bear_researcher",
            "investment_judge",
            "trader",
            "aggressive_debator",
            "conservative_debator",
            "neutral_debator",
            "portfolio_manager",
            "decision_writeback",
        } <= nodes


# ── End-to-end execution tests ───────────────────────────────────────────────


@pytest.mark.unit
@pytest.mark.asyncio
class TestInvestmentDebateGraph:
    async def test_default_rounds_runs_full_debate(self):
        # Default = 2 rounds = 4 debate turns + 1 judge call.
        # Sequence: Bull, Bear, Bull, Bear, Judge.
        responses = [
            ai("bull 1"),
            ai("bear 1"),
            ai("bull 2"),
            ai("bear 2"),
            _judge_output(),
        ]
        patcher = _patch_model_config(_make_role_factory({"sequence": responses}))
        patcher.start()
        try:
            graph = build_investment_debate_graph()
            result = await graph.ainvoke(
                {"analysis_context": {"ticker": "AAPL"}},
                config={"configurable": {"thread_id": "test-debate"}},
            )
        finally:
            patcher.stop()

        # All 4 debate turns accumulated.
        assert result["investment_bull_responses"] == ["bull 1", "bull 2"]
        assert result["investment_bear_responses"] == ["bear 1", "bear 2"]
        # Judge output is set.
        assert result["investment_judge"]["signal"] == "buy"

    async def test_one_round_override(self):
        responses = [ai("bull"), ai("bear"), _judge_output()]
        patcher = _patch_model_config(_make_role_factory({"sequence": responses}))
        patcher.start()
        try:
            graph = build_investment_debate_graph()
            result = await graph.ainvoke(
                {"analysis_context": {"ticker": "AAPL"}},
                config={
                    "configurable": {
                        "thread_id": "test-1round",
                        "max_investment_debate_rounds": 1,
                    }
                },
            )
        finally:
            patcher.stop()

        assert len(result["investment_bull_responses"]) == 1
        assert len(result["investment_bear_responses"]) == 1


@pytest.mark.unit
@pytest.mark.asyncio
class TestInvestmentThesisGraph:
    async def test_runs_through_to_trader(self):
        # 1 round investment debate + judge + trader.
        responses = [
            ai("bull"),
            ai("bear"),
            _judge_output(),
            _trader_output(),
        ]
        patcher = _patch_model_config(_make_role_factory({"sequence": responses}))
        patcher.start()
        try:
            graph = build_investment_thesis_graph()
            result = await graph.ainvoke(
                {"analysis_context": {"ticker": "AAPL"}},
                config={
                    "configurable": {
                        "thread_id": "test-thesis",
                        "max_investment_debate_rounds": 1,
                    }
                },
            )
        finally:
            patcher.stop()

        assert result["investment_judge"]["signal"] == "buy"
        assert result["trader"]["action"] == "buy"


@pytest.mark.unit
@pytest.mark.asyncio
class TestTradingDecisionGraph:
    async def test_runs_full_pipeline_with_reflection_bookends(self):
        # Sequence: bull, bear, judge, trader, agg, cons, neut, PM.
        # (Reflector LLM not called when no prior pending entries.)
        responses = [
            ai("bull"),
            ai("bear"),
            _judge_output(),
            _trader_output(),
            ai("aggressive"),
            ai("conservative"),
            ai("neutral"),
            _pm_output(),
        ]
        patcher = _patch_model_config(_make_role_factory({"sequence": responses}))
        patcher.start()
        try:
            graph = build_trading_decision_graph(store=InMemoryStore())
            result = await graph.ainvoke(
                {"analysis_context": {"ticker": "AAPL"}},
                config={
                    "configurable": {
                        "thread_id": "test-full",
                        "user_id": "alice",
                        "max_investment_debate_rounds": 1,
                        "decision_date": "2026-05-17",
                    }
                },
            )
        finally:
            patcher.stop()

        assert result["portfolio_decision"]["rating"] == "buy"
        assert result["portfolio_decision"]["price_target"] == 220.0
        assert result["decision_date"] == "2026-05-17"

    async def test_two_sequential_runs_inject_reflection(self):
        store = InMemoryStore()

        # Run 1: no prior reflections; reflector_resolve emits no LLM calls.
        # Sequence is: bull, bear, judge, trader, agg, cons, neut, PM.
        run1_responses = [
            ai("bull"),
            ai("bear"),
            _judge_output(),
            _trader_output(),
            ai("agg"),
            ai("cons"),
            ai("neut"),
            _pm_output(),
        ]
        patcher = _patch_model_config(_make_role_factory({"sequence": run1_responses}))
        patcher.start()
        try:
            graph = build_trading_decision_graph(store=store)
            await graph.ainvoke(
                {"analysis_context": {"ticker": "AAPL"}},
                config={
                    "configurable": {
                        "thread_id": "test-run1",
                        "user_id": "alice",
                        "max_investment_debate_rounds": 1,
                        "decision_date": "2026-05-10",
                    }
                },
            )
        finally:
            patcher.stop()

        # Run 1 wrote a pending decision.
        from muffin_agent.agents.trading_decision.reflection.memory import (
            ReflectionMemory,
        )

        memory = ReflectionMemory(store, user_id="alice")
        assert len(await memory.list_pending()) == 1

        # Run 2: same store, new date. Resolver will fetch outcomes for
        # the pending entry and call the reflector. Provide a stub fetcher
        # that returns a real outcome.
        outcome = Outcome(
            raw_return_pct=4.0,
            alpha_return_pct=1.2,
            holding_days=5,
            decision_action="buy",
        )

        async def stub_fetcher(**kwargs: Any) -> Outcome:
            return outcome

        # Sequence: reflector LLM (for resolving prior pending),
        # then bull, bear, judge, trader, agg, cons, neut, PM.
        run2_responses = [
            ai("Bull held; alpha +1.2%."),  # reflector for resolved entry
            ai("bull r2"),
            ai("bear r2"),
            _judge_output(),
            _trader_output(),
            ai("agg r2"),
            ai("cons r2"),
            ai("neut r2"),
            _pm_output(),
        ]
        patcher = _patch_model_config(_make_role_factory({"sequence": run2_responses}))
        patcher.start()
        try:
            graph = build_trading_decision_graph(
                store=store, outcomes_fetcher=stub_fetcher
            )
            result = await graph.ainvoke(
                {"analysis_context": {"ticker": "AAPL"}},
                config={
                    "configurable": {
                        "thread_id": "test-run2",
                        "user_id": "alice",
                        "max_investment_debate_rounds": 1,
                        "decision_date": "2026-05-17",
                    }
                },
            )
        finally:
            patcher.stop()

        # Past reflection from run 1 should be present in the resolved state.
        assert len(result["resolved_decisions"]) == 1
        # The run-1 entry is now resolved in memory.
        resolved = await memory.list_resolved_for_ticker("AAPL")
        assert len(resolved) == 1
        assert resolved[0].reflection == "Bull held; alpha +1.2%."
