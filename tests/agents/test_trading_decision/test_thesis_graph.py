"""End-to-end tests for ``build_investment_thesis_graph`` (PR 2).

Adds a Trader stage downstream of the Investment Judge. Verifies:

* The graph compiles with the expected node set.
* A default-rounds run produces debate → judge → trader and writes both
  ``investment_judge`` and ``trader`` to the final state.
* Trader fallback survives a missing structured response.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.messages import AIMessage

from muffin_agent.agents.trading_decision import (
    AnalysisContext,
    InvestmentJudgeOutput,
    TraderOutput,
    build_investment_thesis_graph,
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
        "reasoning": "Judge conviction supports a starter long.",
        "entry_price": 195.0,
        "stop_loss": 180.0,
        "take_profit": 225.0,
        "position_sizing": "2% of NAV starter",
        "time_horizon": "3–6 months",
    }
    base.update(overrides)
    return base


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


def _patch_factories(*, judge_payload=None, trader_payload=None, trader_raises=False):
    bull_agent = AsyncMock()
    bull_agent.ainvoke.return_value = {"messages": [AIMessage(content="Bull arg.")]}
    bear_agent = AsyncMock()
    bear_agent.ainvoke.return_value = {"messages": [AIMessage(content="Bear arg.")]}
    judge_agent = AsyncMock()
    judge_agent.ainvoke.return_value = _judge_result(judge_payload or _judge_payload())
    trader_agent = AsyncMock()
    if trader_raises:
        trader_agent.ainvoke.side_effect = RuntimeError("trader failed")
    else:
        trader_agent.ainvoke.return_value = _trader_result(
            trader_payload or _trader_payload()
        )

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
        ],
        bull_agent,
        bear_agent,
        judge_agent,
        trader_agent,
    )


@pytest.mark.unit
class TestBuildInvestmentThesisGraph:
    def test_graph_compiles_with_trader(self):
        graph = build_investment_thesis_graph()
        node_names = set(graph.get_graph().nodes.keys())
        assert {
            "bull_researcher",
            "bear_researcher",
            "investment_judge",
            "trader",
        } <= node_names


@pytest.mark.unit
@pytest.mark.asyncio
class TestInvestmentThesisExecution:
    async def test_default_run_produces_trader_output(self):
        patches, _, _, _, trader_agent = _patch_factories()
        for cm in patches:
            cm.start()
        try:
            graph = build_investment_thesis_graph()
            ctx = AnalysisContext.from_narrative("AAPL", "Notes.")
            final_state = await graph.ainvoke(
                {"analysis_context": ctx.model_dump()},
                config={"configurable": {"thread_id": "thesis-test-default"}},
            )
        finally:
            for cm in patches:
                cm.stop()

        # Judge ran once, Trader ran once.
        assert trader_agent.ainvoke.await_count == 1

        # Judge output present.
        judge = final_state["investment_judge"]
        assert judge["signal"] == "buy"

        # Trader output present and structured.
        trader = final_state["trader"]
        assert trader["action"] == "buy"
        assert trader["entry_price"] == 195.0
        assert trader["stop_loss"] == 180.0
        assert "error" not in trader

    async def test_trader_failure_does_not_break_pipeline(self):
        patches, _, _, _, _ = _patch_factories(trader_raises=True)
        for cm in patches:
            cm.start()
        try:
            graph = build_investment_thesis_graph()
            ctx = AnalysisContext.from_narrative("AAPL", "Notes.")
            final_state = await graph.ainvoke(
                {"analysis_context": ctx.model_dump()},
                config={"configurable": {"thread_id": "thesis-test-fail"}},
            )
        finally:
            for cm in patches:
                cm.stop()

        # Judge output still made it through.
        assert final_state["investment_judge"]["signal"] == "buy"
        # Trader fallback recorded.
        assert final_state["trader"]["action"] == "hold"
        assert "raised" in final_state["trader"]["error"].lower()
