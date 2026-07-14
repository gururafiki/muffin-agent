"""End-to-end tests for the three trading_decision graph builders.

The 4 analyst agents (Market / Fundamentals / News / Social) are ReAct
agents that fetch data via OpenBB MCP — far too heavyweight to drive
through with a fake LLM in unit tests. The tests patch
``_add_analyst_nodes`` to register lightweight stub nodes that just
write constant report strings into state. This isolates the
debate/judge/trader/risk/PM/reflection logic from analyst behaviour;
the analyst agents themselves are covered by their own unit tests
(when added).
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest
from langgraph.graph import StateGraph
from langgraph.store.memory import InMemoryStore
from langgraph.types import RetryPolicy

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

_GRAPH_MODULE = "muffin_agent.agents.trading_decision.graph"


# ── Fixtures: structured outputs the synthesis nodes consume ─────────────────


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


# ── Fake LLM sequencing ──────────────────────────────────────────────────────


def _make_role_factory(role_responses: dict[str, Any]):
    """Build a fake ``ModelConfiguration.from_runnable_config``.

    Each ``get_llm_for_role`` call returns a FRESH FakeLLM bound to the
    next response in the queue. All non-analyst LLM-call nodes share the
    ``"reasoner"`` role and step through a single queue.
    """
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
    """Patch ``ModelConfiguration.from_runnable_config`` at the source."""
    from muffin_agent.model_config import ModelConfiguration

    return patch.object(ModelConfiguration, "from_runnable_config", side_effect=factory)


# ── Stub analyst nodes ───────────────────────────────────────────────────────


_LLM_RETRY = RetryPolicy(max_attempts=2)


async def _stub_market(state):  # noqa: ARG001
    return {"market_report": "Market: trend up, RSI 58."}


async def _stub_fundamentals(state):  # noqa: ARG001
    return {"fundamentals_report": "Fund: ROIC 28%, FCF margin 22%."}


async def _stub_news(state):  # noqa: ARG001
    return {"news_report": "News: Q1 earnings 2026-04-25."}


async def _stub_social(state):  # noqa: ARG001
    return {"sentiment_report": "Sentiment: retail bullish."}


async def _stub_add_analyst_nodes(
    graph: StateGraph,
    config,  # noqa: ARG001
) -> None:
    """Replacement for ``_add_analyst_nodes`` that wires lightweight stubs.

    The real analyst agents make MCP calls + multi-step ReAct loops,
    which are infeasible to drive end-to-end with a FakeLLM. We replace
    them with single-function nodes that emit constant report strings.
    """
    graph.add_node("market_analyst", _stub_market, retry_policy=_LLM_RETRY)
    graph.add_node("fundamentals_analyst", _stub_fundamentals, retry_policy=_LLM_RETRY)
    graph.add_node("news_analyst", _stub_news, retry_policy=_LLM_RETRY)
    graph.add_node("social_analyst", _stub_social, retry_policy=_LLM_RETRY)


def _patch_analyst_nodes():
    return patch(f"{_GRAPH_MODULE}._add_analyst_nodes", new=_stub_add_analyst_nodes)


# ── Graph compilation tests ──────────────────────────────────────────────────


@pytest.mark.unit
@pytest.mark.asyncio
class TestGraphCompilation:
    async def test_investment_debate_graph_compiles(self):
        with _patch_analyst_nodes():
            g = await build_investment_debate_graph(config={"configurable": {}})
        nodes = set(g.get_graph().nodes.keys())
        # Bull/Bear now live inside the `investment_debate` conference
        # subgraph, not as top-level nodes.
        assert {
            "market_analyst",
            "fundamentals_analyst",
            "news_analyst",
            "social_analyst",
            "investment_debate",
            "investment_judge",
        } <= nodes
        assert "bull_researcher" not in nodes
        assert "bear_researcher" not in nodes

    async def test_investment_thesis_graph_compiles(self):
        with _patch_analyst_nodes():
            g = await build_investment_thesis_graph(config={"configurable": {}})
        nodes = set(g.get_graph().nodes.keys())
        assert {
            "market_analyst",
            "investment_debate",
            "investment_judge",
            "trader",
        } <= nodes

    async def test_trading_decision_graph_compiles(self):
        with _patch_analyst_nodes():
            g = await build_trading_decision_graph(config={"configurable": {}})
        nodes = set(g.get_graph().nodes.keys())
        # Both debates now live inside their conference subgraphs
        # (`investment_debate` / `risk_debate`), not as top-level nodes.
        assert {
            "reflector_resolve",
            "market_analyst",
            "fundamentals_analyst",
            "news_analyst",
            "social_analyst",
            "investment_debate",
            "investment_judge",
            "trader",
            "risk_debate",
            "portfolio_manager",
            "decision_writeback",
        } <= nodes
        assert "bull_researcher" not in nodes
        assert "bear_researcher" not in nodes


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
            with _patch_analyst_nodes():
                graph = await build_investment_debate_graph(config={"configurable": {}})
                result = await graph.ainvoke(
                    {"ticker": "AAPL"},
                    config={"configurable": {"thread_id": "test-debate"}},
                )
        finally:
            patcher.stop()

        # The Bull/Bear debate is now a conference subgraph — turns land as
        # name-tagged AIMessages in `investment_debate_messages`, exactly one
        # per turn (no reducer echo / duplication).
        msgs = result["investment_debate_messages"]
        assert [m.name for m in msgs] == [
            "bull_researcher",
            "bear_researcher",
            "bull_researcher",
            "bear_researcher",
        ]
        assert [m.content for m in msgs] == ["bull 1", "bear 1", "bull 2", "bear 2"]
        assert result["investment_judge"]["signal"] == "buy"
        # Analyst stubs wrote their reports too.
        assert result["market_report"].startswith("Market:")
        assert result["fundamentals_report"].startswith("Fund:")

    async def test_one_round_override(self):
        responses = [ai("bull"), ai("bear"), _judge_output()]
        patcher = _patch_model_config(_make_role_factory({"sequence": responses}))
        patcher.start()
        # Debate rounds size the conference subgraph at BUILD time (same as
        # the risk debate + the CLI, which reuses one config for build+invoke).
        run_config = {
            "configurable": {
                "thread_id": "test-1round",
                "max_investment_debate_rounds": 1,
            }
        }
        try:
            with _patch_analyst_nodes():
                graph = await build_investment_debate_graph(config=run_config)
                result = await graph.ainvoke({"ticker": "AAPL"}, config=run_config)
        finally:
            patcher.stop()

        msgs = result["investment_debate_messages"]
        assert len(msgs) == 2
        assert [m.name for m in msgs] == ["bull_researcher", "bear_researcher"]


@pytest.mark.unit
@pytest.mark.asyncio
class TestInvestmentThesisGraph:
    async def test_runs_through_to_trader(self):
        responses = [
            ai("bull"),
            ai("bear"),
            _judge_output(),
            _trader_output(),
        ]
        patcher = _patch_model_config(_make_role_factory({"sequence": responses}))
        patcher.start()
        run_config = {
            "configurable": {
                "thread_id": "test-thesis",
                "max_investment_debate_rounds": 1,
            }
        }
        try:
            with _patch_analyst_nodes():
                graph = await build_investment_thesis_graph(config=run_config)
                result = await graph.ainvoke({"ticker": "AAPL"}, config=run_config)
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
        run_config = {
            "configurable": {
                "thread_id": "test-full",
                "user_id": "alice",
                "max_investment_debate_rounds": 1,
                "decision_date": "2026-05-17",
            }
        }
        try:
            with _patch_analyst_nodes():
                graph = await build_trading_decision_graph(
                    config=run_config,
                    store=InMemoryStore(),
                )
                result = await graph.ainvoke({"ticker": "AAPL"}, config=run_config)
        finally:
            patcher.stop()

        assert result["portfolio_decision"]["rating"] == "buy"
        assert result["portfolio_decision"]["price_target"] == 220.0
        assert result["decision_date"] == "2026-05-17"
        # Regression (bug: bull/bear turns doubled 2→4). risk_debate runs
        # AFTER investment_debate and shares the parent state schema; before
        # the `output_schema` restriction it echoed the investment-debate
        # channel back through write-back, and the parent reducer re-applied
        # it. Assert exactly the turns each conference produced — no echo.
        assert len(result["investment_debate_messages"]) == 2  # 1 round × 2
        assert len(result["risk_debate_messages"]) == 3  # 1 round × 3

    async def test_two_sequential_runs_inject_reflection(self):
        store = InMemoryStore()

        # Run 1: bull, bear, judge, trader, agg, cons, neut, PM.
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
        run1_config = {
            "configurable": {
                "thread_id": "test-run1",
                "user_id": "alice",
                "max_investment_debate_rounds": 1,
                "decision_date": "2026-05-10",
            }
        }
        try:
            with _patch_analyst_nodes():
                graph = await build_trading_decision_graph(
                    config=run1_config, store=store
                )
                await graph.ainvoke({"ticker": "AAPL"}, config=run1_config)
        finally:
            patcher.stop()

        from muffin_agent.agents.trading_decision.reflection.memory import (
            ReflectionMemory,
        )

        memory = ReflectionMemory(store, user_id="alice")
        assert len(await memory.list_pending()) == 1

        # Run 2: reflector LLM resolves prior, then the standard 8 calls.
        outcome = Outcome(
            raw_return_pct=4.0,
            alpha_return_pct=1.2,
            holding_days=5,
            decision_action="buy",
        )

        async def stub_fetcher(**kwargs: Any) -> Outcome:  # noqa: ARG001
            return outcome

        run2_responses = [
            ai("Bull held; alpha +1.2%."),
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
        run2_config = {
            "configurable": {
                "thread_id": "test-run2",
                "user_id": "alice",
                "max_investment_debate_rounds": 1,
                "decision_date": "2026-05-17",
            }
        }
        try:
            with _patch_analyst_nodes():
                graph = await build_trading_decision_graph(
                    config=run2_config,
                    store=store,
                    outcomes_fetcher=stub_fetcher,
                )
                result = await graph.ainvoke({"ticker": "AAPL"}, config=run2_config)
        finally:
            patcher.stop()

        assert len(result["resolved_decisions"]) == 1
        resolved = await memory.list_resolved_for_ticker("AAPL")
        assert len(resolved) == 1
        assert resolved[0].reflection == "Bull held; alpha +1.2%."
