"""End-to-end tests for ``build_trading_decision_graph`` (PR 3).

Patches all 8 agent factories with deterministic fakes and verifies the
full pipeline runs debate → judge → trader → risk debate → portfolio
manager. The Portfolio Manager's output is the canonical artifact.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.messages import AIMessage

from muffin_agent.agents.trading_decision import (
    AnalysisContext,
    InvestmentJudgeOutput,
    PortfolioDecisionOutput,
    TraderOutput,
    build_trading_decision_graph,
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


def _trader_payload(**overrides) -> dict:
    base = {
        "action": "buy",
        "reasoning": "Judge supports starter long.",
        "entry_price": 195.0,
        "stop_loss": 180.0,
        "take_profit": 225.0,
        "position_sizing": "2% of NAV starter",
        "time_horizon": "3–6 months",
    }
    base.update(overrides)
    return base


def _pm_payload(**overrides) -> dict:
    base = {
        "rating": "buy",
        "executive_summary": "Buy AAPL at 2% NAV; primary risk China demand.",
        "investment_thesis": "Bull case held; Trader stop tightened after debate.",
        "price_target": 220.0,
        "stop_loss": 182.5,
        "time_horizon": "3–6 months",
        "position_sizing": "2% of NAV starter, scale to 4% on Q1 beat",
        "key_risks_remaining": ["China demand"],
        "confidence": 0.65,
    }
    base.update(overrides)
    return base


def _ai(text: str) -> dict:
    return {"messages": [AIMessage(content=text)]}


def _judge_result(payload: dict) -> dict:
    return {
        "messages": [AIMessage(content="judge ok")],
        "structured_response": InvestmentJudgeOutput.model_validate(payload),
    }


def _trader_result(payload: dict) -> dict:
    return {
        "messages": [AIMessage(content="trader ok")],
        "structured_response": TraderOutput.model_validate(payload),
    }


def _pm_result(payload: dict) -> dict:
    return {
        "messages": [AIMessage(content="pm ok")],
        "structured_response": PortfolioDecisionOutput.model_validate(payload),
    }


def _patch_all_factories(
    *,
    pm_payload: dict | None = None,
    pm_raises: bool = False,
):
    bull_agent = AsyncMock()
    bull_agent.ainvoke.return_value = _ai("Bull arg.")
    bear_agent = AsyncMock()
    bear_agent.ainvoke.return_value = _ai("Bear arg.")
    judge_agent = AsyncMock()
    judge_agent.ainvoke.return_value = _judge_result(_judge_payload())
    trader_agent = AsyncMock()
    trader_agent.ainvoke.return_value = _trader_result(_trader_payload())

    aggressive_agent = AsyncMock()
    aggressive_agent.ainvoke.return_value = _ai("Press it — upside dominates.")
    conservative_agent = AsyncMock()
    conservative_agent.ainvoke.return_value = _ai("Stop is tight — China demand real.")
    neutral_agent = AsyncMock()
    neutral_agent.ainvoke.return_value = _ai("Scale in via starter + catalyst add.")

    pm_agent = AsyncMock()
    if pm_raises:
        pm_agent.ainvoke.side_effect = RuntimeError("PM failed")
    else:
        pm_agent.ainvoke.return_value = _pm_result(pm_payload or _pm_payload())

    return (
        [
            patch(
                "muffin_agent.agents.trading_decision.nodes.create_bull_researcher_agent",
                AsyncMock(return_value=bull_agent),
            ),
            patch(
                "muffin_agent.agents.trading_decision.nodes.create_bear_researcher_agent",
                AsyncMock(return_value=bear_agent),
            ),
            patch(
                "muffin_agent.agents.trading_decision.nodes.create_investment_judge_agent",
                AsyncMock(return_value=judge_agent),
            ),
            patch(
                "muffin_agent.agents.trading_decision.nodes.create_trader_agent",
                AsyncMock(return_value=trader_agent),
            ),
            patch(
                "muffin_agent.agents.trading_decision.nodes.create_aggressive_debator_agent",
                AsyncMock(return_value=aggressive_agent),
            ),
            patch(
                "muffin_agent.agents.trading_decision.nodes.create_conservative_debator_agent",
                AsyncMock(return_value=conservative_agent),
            ),
            patch(
                "muffin_agent.agents.trading_decision.nodes.create_neutral_debator_agent",
                AsyncMock(return_value=neutral_agent),
            ),
            patch(
                "muffin_agent.agents.trading_decision.nodes.create_portfolio_manager_agent",
                AsyncMock(return_value=pm_agent),
            ),
        ],
        {
            "bull": bull_agent,
            "bear": bear_agent,
            "judge": judge_agent,
            "trader": trader_agent,
            "aggressive": aggressive_agent,
            "conservative": conservative_agent,
            "neutral": neutral_agent,
            "pm": pm_agent,
        },
    )


@pytest.mark.unit
class TestBuildTradingDecisionGraph:
    def test_graph_compiles_with_all_nodes(self):
        graph = build_trading_decision_graph()
        node_names = set(graph.get_graph().nodes.keys())
        assert {
            "bull_researcher",
            "bear_researcher",
            "investment_judge",
            "trader",
            "aggressive_debator",
            "conservative_debator",
            "neutral_debator",
            "portfolio_manager",
        } <= node_names


@pytest.mark.unit
@pytest.mark.asyncio
class TestTradingDecisionExecution:
    async def test_default_run_produces_canonical_portfolio_decision(self):
        patches, agents = _patch_all_factories()
        for cm in patches:
            cm.start()
        try:
            graph = build_trading_decision_graph()
            ctx = AnalysisContext.from_narrative("AAPL", "AAPL notes.")
            final_state = await graph.ainvoke(
                {"analysis_context": ctx.model_dump()},
                config={"configurable": {"thread_id": "full-default"}},
            )
        finally:
            for cm in patches:
                cm.stop()

        # Investment debate: default 2 rounds = 4 turns.
        assert agents["bull"].ainvoke.await_count == 2
        assert agents["bear"].ainvoke.await_count == 2
        # Judge + Trader run once each.
        assert agents["judge"].ainvoke.await_count == 1
        assert agents["trader"].ainvoke.await_count == 1
        # Risk debate: default 1 round = 3 turns (one per debater).
        assert agents["aggressive"].ainvoke.await_count == 1
        assert agents["conservative"].ainvoke.await_count == 1
        assert agents["neutral"].ainvoke.await_count == 1
        # Portfolio Manager runs once.
        assert agents["pm"].ainvoke.await_count == 1

        # Canonical artifact populated.
        decision = final_state["portfolio_decision"]
        assert decision["rating"] == "buy"
        assert decision["price_target"] == 220.0
        assert decision["confidence"] == 0.65
        assert "error" not in decision

        # Risk debate transcript interleaves all three speakers.
        risk_history = final_state["risk_debate"]["history"]
        assert "Aggressive Analyst:" in risk_history
        assert "Conservative Analyst:" in risk_history
        assert "Neutral Analyst:" in risk_history

    async def test_two_rounds_risk_debate_runs_6_turns(self):
        patches, agents = _patch_all_factories()
        for cm in patches:
            cm.start()
        try:
            graph = build_trading_decision_graph()
            ctx = AnalysisContext.from_narrative("AAPL", "Notes.")
            await graph.ainvoke(
                {"analysis_context": ctx.model_dump()},
                config={
                    "configurable": {
                        "thread_id": "full-2rounds",
                        "max_risk_debate_rounds": 2,
                    }
                },
            )
        finally:
            for cm in patches:
                cm.stop()

        # Each risk debater should run twice.
        assert agents["aggressive"].ainvoke.await_count == 2
        assert agents["conservative"].ainvoke.await_count == 2
        assert agents["neutral"].ainvoke.await_count == 2

    async def test_strong_sell_decision_propagates(self):
        patches, _ = _patch_all_factories(
            pm_payload=_pm_payload(
                rating="strong_sell",
                executive_summary="Exit AAPL fully.",
                confidence=0.85,
            ),
        )
        for cm in patches:
            cm.start()
        try:
            graph = build_trading_decision_graph()
            ctx = AnalysisContext.from_narrative("AAPL", "Notes.")
            final_state = await graph.ainvoke(
                {"analysis_context": ctx.model_dump()},
                config={"configurable": {"thread_id": "full-strong-sell"}},
            )
        finally:
            for cm in patches:
                cm.stop()

        decision = final_state["portfolio_decision"]
        assert decision["rating"] == "strong_sell"
        assert decision["confidence"] == 0.85

    async def test_pm_failure_yields_fallback_but_pipeline_completes(self):
        patches, _ = _patch_all_factories(pm_raises=True)
        for cm in patches:
            cm.start()
        try:
            graph = build_trading_decision_graph()
            ctx = AnalysisContext.from_narrative("AAPL", "Notes.")
            final_state = await graph.ainvoke(
                {"analysis_context": ctx.model_dump()},
                config={"configurable": {"thread_id": "full-pm-fail"}},
            )
        finally:
            for cm in patches:
                cm.stop()

        # All earlier outputs still present.
        assert final_state["investment_judge"]["signal"] == "buy"
        assert final_state["trader"]["action"] == "buy"
        # PM fell back gracefully.
        assert final_state["portfolio_decision"]["rating"] == "hold"
        assert "raised" in final_state["portfolio_decision"]["error"].lower()
