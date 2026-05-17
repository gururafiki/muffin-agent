"""End-to-end tests for the reflection-memory loop (PR 4).

Exercises two sequential ``build_trading_decision_graph`` runs that share
an ``InMemoryStore``:

* Run 1: no prior reflections → PM prompt has empty ``past_reflections``;
  writeback creates a new pending entry.
* Run 2: pending entry is resolved with a stub outcome + stub reflection;
  PM prompt receives the resolved reflection block; writeback creates the
  next pending entry.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.messages import AIMessage
from langgraph.store.memory import InMemoryStore

from muffin_agent.agents.trading_decision import (
    AnalysisContext,
    InvestmentJudgeOutput,
    Outcome,
    PortfolioDecisionOutput,
    ReflectionMemory,
    TraderOutput,
    build_trading_decision_graph,
)


def _judge_payload(**overrides) -> dict:
    base = {
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
    base.update(overrides)
    return base


def _trader_payload(**overrides) -> dict:
    base = {
        "action": "buy",
        "reasoning": "Starter long.",
        "entry_price": 195.0,
        "stop_loss": 180.0,
        "take_profit": 225.0,
        "position_sizing": "2% NAV",
        "time_horizon": "3-6m",
    }
    base.update(overrides)
    return base


def _pm_payload(**overrides) -> dict:
    base = {
        "rating": "buy",
        "executive_summary": "Buy AAPL.",
        "investment_thesis": "Bull held after risk debate.",
        "price_target": 220.0,
        "stop_loss": 180.0,
        "time_horizon": "3-6m",
        "position_sizing": "2% NAV starter",
        "key_risks_remaining": ["China"],
        "confidence": 0.7,
        "incorporates_past_lessons": False,
    }
    base.update(overrides)
    return base


def _stub_factories(
    *,
    captured_pm_prompts: list[str] | None = None,
    pm_payload: dict | None = None,
):
    """Build deterministic AsyncMock factories for all 8 agents.

    Optionally captures the system prompt the Portfolio Manager was built
    with into ``captured_pm_prompts`` (so reflection-injection tests can
    assert the past-reflections block was included).
    """
    bull_agent = AsyncMock()
    bull_agent.ainvoke.return_value = {"messages": [AIMessage(content="bull")]}
    bear_agent = AsyncMock()
    bear_agent.ainvoke.return_value = {"messages": [AIMessage(content="bear")]}
    judge_agent = AsyncMock()
    judge_agent.ainvoke.return_value = {
        "messages": [AIMessage(content="judge")],
        "structured_response": InvestmentJudgeOutput.model_validate(_judge_payload()),
    }
    trader_agent = AsyncMock()
    trader_agent.ainvoke.return_value = {
        "messages": [AIMessage(content="trader")],
        "structured_response": TraderOutput.model_validate(_trader_payload()),
    }
    agg = AsyncMock()
    agg.ainvoke.return_value = {"messages": [AIMessage(content="agg")]}
    cons = AsyncMock()
    cons.ainvoke.return_value = {"messages": [AIMessage(content="cons")]}
    neut = AsyncMock()
    neut.ainvoke.return_value = {"messages": [AIMessage(content="neut")]}
    pm_agent = AsyncMock()
    pm_agent.ainvoke.return_value = {
        "messages": [AIMessage(content="pm")],
        "structured_response": PortfolioDecisionOutput.model_validate(
            pm_payload or _pm_payload()
        ),
    }

    bull_factory = AsyncMock(return_value=bull_agent)
    bear_factory = AsyncMock(return_value=bear_agent)
    judge_factory = AsyncMock(return_value=judge_agent)
    trader_factory = AsyncMock(return_value=trader_agent)
    agg_factory = AsyncMock(return_value=agg)
    cons_factory = AsyncMock(return_value=cons)
    neut_factory = AsyncMock(return_value=neut)
    pm_factory = AsyncMock(return_value=pm_agent)

    if captured_pm_prompts is not None:
        # Wrap the PM factory to capture the past_reflections arg every call.
        original = pm_factory

        async def capturing_pm_factory(
            config: Any,
            *,
            ticker: str,
            query: str | None,
            context_vars: dict,
            investment_judge: dict,
            trader: dict,
            risk_debate_history: str,
            past_reflections: str = "",
        ):
            captured_pm_prompts.append(past_reflections)
            return await original(
                config,
                ticker=ticker,
                query=query,
                context_vars=context_vars,
                investment_judge=investment_judge,
                trader=trader,
                risk_debate_history=risk_debate_history,
                past_reflections=past_reflections,
            )

        pm_factory = capturing_pm_factory  # type: ignore[assignment]

    patches = [
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
        patch(
            "muffin_agent.agents.trading_decision.nodes.create_trader_agent",
            trader_factory,
        ),
        patch(
            "muffin_agent.agents.trading_decision.nodes.create_aggressive_debator_agent",
            agg_factory,
        ),
        patch(
            "muffin_agent.agents.trading_decision.nodes.create_conservative_debator_agent",
            cons_factory,
        ),
        patch(
            "muffin_agent.agents.trading_decision.nodes.create_neutral_debator_agent",
            neut_factory,
        ),
        patch(
            "muffin_agent.agents.trading_decision.nodes.create_portfolio_manager_agent",
            pm_factory,
        ),
    ]
    return patches


def _stub_outcome_fetcher(outcome: Outcome | None):
    async def fetcher(**kwargs: Any) -> Outcome | None:
        return outcome

    return fetcher


@pytest.mark.unit
@pytest.mark.asyncio
class TestReflectionLoop:
    async def test_writeback_then_resolve_then_inject_across_runs(self):
        store = InMemoryStore()
        captured_prompts: list[str] = []
        patches = _stub_factories(captured_pm_prompts=captured_prompts)
        outcome = Outcome(
            raw_return_pct=4.0,
            alpha_return_pct=1.2,
            holding_days=5,
            decision_action="buy",
        )

        for cm in patches:
            cm.start()
        try:
            graph = build_trading_decision_graph(
                store=store,
                outcomes_fetcher=_stub_outcome_fetcher(outcome),
            )
            ctx = AnalysisContext.from_narrative("AAPL", "Notes.")

            # Run 1 — first ever decision for this user/ticker.
            with patch(
                "muffin_agent.agents.trading_decision.nodes.generate_reflection",
                AsyncMock(return_value="Bull thesis held; alpha +1.2%."),
            ):
                run1 = await graph.ainvoke(
                    {"analysis_context": ctx.model_dump()},
                    config={
                        "configurable": {
                            "thread_id": "reflection-test-1",
                            "user_id": "alice",
                            "decision_date": "2026-05-10",
                        }
                    },
                )

            # Run 1 PM prompt had no prior reflections.
            assert len(captured_prompts) == 1
            assert captured_prompts[0] == ""

            # Run 1 wrote a pending decision.
            memory = ReflectionMemory(store, user_id="alice")
            assert len(await memory.list_pending()) == 1
            # And run1 produced a PortfolioDecision.
            assert run1["portfolio_decision"]["rating"] == "buy"

            # Run 2 — fresh date, same ticker. Should resolve run 1's pending.
            with patch(
                "muffin_agent.agents.trading_decision.nodes.generate_reflection",
                AsyncMock(return_value="Bull thesis held; alpha +1.2%."),
            ):
                run2 = await graph.ainvoke(
                    {"analysis_context": ctx.model_dump()},
                    config={
                        "configurable": {
                            "thread_id": "reflection-test-2",
                            "user_id": "alice",
                            "decision_date": "2026-05-17",
                        }
                    },
                )

            # Run 2 PM prompt should contain the resolved reflection.
            assert len(captured_prompts) == 2
            run2_prompt = captured_prompts[1]
            assert "AAPL 2026-05-10" in run2_prompt
            assert "alpha +1.20%" in run2_prompt
            assert "Bull thesis held" in run2_prompt

            # Resolved decisions surfaced in state for observability.
            assert len(run2["resolved_decisions"]) == 1
            assert run2["resolved_decisions"][0]["ticker"] == "AAPL"

            # Run 2 wrote its own pending entry (and run 1 is now resolved).
            pending = await memory.list_pending()
            assert len(pending) == 1
            assert pending[0].date == "2026-05-17"
            resolved = await memory.list_resolved_for_ticker("AAPL")
            assert len(resolved) == 1
            assert resolved[0].date == "2026-05-10"

        finally:
            for cm in patches:
                cm.stop()

    async def test_reflection_disabled_skips_writeback_and_resolve(self):
        store = InMemoryStore()
        captured_prompts: list[str] = []
        patches = _stub_factories(captured_pm_prompts=captured_prompts)

        for cm in patches:
            cm.start()
        try:
            graph = build_trading_decision_graph(store=store)
            ctx = AnalysisContext.from_narrative("AAPL", "Notes.")
            await graph.ainvoke(
                {"analysis_context": ctx.model_dump()},
                config={
                    "configurable": {
                        "thread_id": "reflection-off-1",
                        "user_id": "alice",
                        "reflection_enabled": False,
                        "decision_date": "2026-05-10",
                    }
                },
            )
        finally:
            for cm in patches:
                cm.stop()

        assert captured_prompts[0] == ""
        memory = ReflectionMemory(store, user_id="alice")
        assert await memory.list_pending() == []

    async def test_pipeline_works_without_store(self):
        captured_prompts: list[str] = []
        patches = _stub_factories(captured_pm_prompts=captured_prompts)

        for cm in patches:
            cm.start()
        try:
            graph = build_trading_decision_graph(store=None)
            ctx = AnalysisContext.from_narrative("AAPL", "Notes.")
            result = await graph.ainvoke(
                {"analysis_context": ctx.model_dump()},
                config={"configurable": {"thread_id": "no-store"}},
            )
        finally:
            for cm in patches:
                cm.stop()

        # PM ran, decision produced, no reflections.
        assert result["portfolio_decision"]["rating"] == "buy"
        assert captured_prompts[0] == ""
