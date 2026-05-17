"""Tests for the risk debate + Portfolio Manager node wrappers (PR 3)."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.messages import AIMessage

from muffin_agent.agents.trading_decision.conditional_logic import (
    AGGRESSIVE_TAG,
    CONSERVATIVE_TAG,
    NEUTRAL_TAG,
)
from muffin_agent.agents.trading_decision.nodes import (
    aggressive_debator_node,
    conservative_debator_node,
    neutral_debator_node,
    portfolio_manager_node,
)
from muffin_agent.agents.trading_decision.schemas import (
    AnalysisContext,
    PortfolioDecisionOutput,
)


def _ai(text: str) -> dict:
    return {"messages": [AIMessage(content=text)]}


def _judge() -> dict:
    return {
        "signal": "buy",
        "conviction": 0.7,
        "summary": "Net bull.",
        "bull_case": "Bull case.",
        "bear_case": "Bear case.",
        "key_catalysts": ["catalyst"],
        "key_risks": ["risk"],
        "monitoring_checklist": ["metric"],
        "winning_side": "bull",
        "reasoning": "reasoning",
    }


def _trader() -> dict:
    return {
        "action": "buy",
        "reasoning": "starter long",
        "entry_price": 195.0,
        "stop_loss": 180.0,
        "take_profit": 225.0,
        "position_sizing": "2% of NAV",
        "time_horizon": "3–6 months",
    }


def _ctx() -> AnalysisContext:
    return AnalysisContext(ticker="AAPL", narrative="AAPL trades at 22x P/E.")


def _empty_risk_debate(**overrides) -> dict:
    base = {
        "history": "",
        "aggressive_history": "",
        "conservative_history": "",
        "neutral_history": "",
        "current_aggressive_response": "",
        "current_conservative_response": "",
        "current_neutral_response": "",
        "latest_speaker": "",
        "judge_decision": "",
        "count": 0,
    }
    base.update(overrides)
    return base


def _pm_payload(**overrides) -> dict:
    base = {
        "rating": "buy",
        "executive_summary": "Buy AAPL at 2% NAV.",
        "investment_thesis": "Bull case held after debate.",
        "price_target": 220.0,
        "stop_loss": 180.0,
        "time_horizon": "3–6 months",
        "position_sizing": "2% of NAV starter",
        "key_risks_remaining": ["China demand"],
        "confidence": 0.65,
    }
    base.update(overrides)
    return base


def _pm_result(payload: dict) -> dict:
    return {
        "messages": [AIMessage(content="(structured pm response)")],
        "structured_response": PortfolioDecisionOutput.model_validate(payload),
    }


# ── Risk debate nodes ────────────────────────────────────────────────────────


@pytest.mark.unit
@pytest.mark.asyncio
class TestAggressiveDebatorNode:
    async def test_opening_turn_tags_and_increments(self):
        agent = AsyncMock()
        agent.ainvoke.return_value = _ai("Press the position — upside is asymmetric.")

        with patch(
            "muffin_agent.agents.trading_decision.nodes.create_aggressive_debator_agent",
            AsyncMock(return_value=agent),
        ):
            state: dict[str, Any] = {
                "analysis_context": _ctx(),
                "investment_judge": _judge(),
                "trader": _trader(),
            }
            update = await aggressive_debator_node(state, {})

        debate = update["risk_debate"]
        assert debate["latest_speaker"] == "Aggressive"
        assert debate["count"] == 1
        assert debate["current_aggressive_response"].startswith(AGGRESSIVE_TAG)
        assert "asymmetric" in debate["current_aggressive_response"]
        assert debate["aggressive_history"].endswith("asymmetric.")
        assert debate["conservative_history"] == ""
        assert debate["neutral_history"] == ""

    async def test_skips_when_judge_missing(self):
        factory = AsyncMock()
        with patch(
            "muffin_agent.agents.trading_decision.nodes.create_aggressive_debator_agent",
            factory,
        ):
            state = {"analysis_context": _ctx(), "trader": _trader()}
            update = await aggressive_debator_node(state, {})

        assert factory.await_count == 0
        debate = update["risk_debate"]
        assert debate["count"] == 1
        assert "skipped" in debate["current_aggressive_response"].lower()
        assert debate["latest_speaker"] == "Aggressive"

    async def test_skips_when_trader_errored(self):
        factory = AsyncMock()
        with patch(
            "muffin_agent.agents.trading_decision.nodes.create_aggressive_debator_agent",
            factory,
        ):
            state = {
                "analysis_context": _ctx(),
                "investment_judge": _judge(),
                "trader": {"action": "hold", "error": "trader failed"},
            }
            update = await aggressive_debator_node(state, {})

        assert factory.await_count == 0
        assert update["risk_debate"]["count"] == 1
        assert "skipped" in update["risk_debate"]["current_aggressive_response"].lower()

    async def test_exception_yields_failure_marker(self):
        agent = AsyncMock()
        agent.ainvoke.side_effect = RuntimeError("LLM failure")

        with patch(
            "muffin_agent.agents.trading_decision.nodes.create_aggressive_debator_agent",
            AsyncMock(return_value=agent),
        ):
            state = {
                "analysis_context": _ctx(),
                "investment_judge": _judge(),
                "trader": _trader(),
            }
            update = await aggressive_debator_node(state, {})

        debate = update["risk_debate"]
        assert debate["count"] == 1
        assert "failed" in debate["current_aggressive_response"].lower()
        assert debate["latest_speaker"] == "Aggressive"


@pytest.mark.unit
@pytest.mark.asyncio
class TestConservativeDebatorNode:
    async def test_preserves_aggressive_history(self):
        agent = AsyncMock()
        agent.ainvoke.return_value = _ai("Stop is too tight — vol is elevated.")

        with patch(
            "muffin_agent.agents.trading_decision.nodes.create_conservative_debator_agent",
            AsyncMock(return_value=agent),
        ):
            state = {
                "analysis_context": _ctx(),
                "investment_judge": _judge(),
                "trader": _trader(),
                "risk_debate": _empty_risk_debate(
                    history="Aggressive Analyst: Press it.",
                    aggressive_history="Aggressive Analyst: Press it.",
                    current_aggressive_response="Aggressive Analyst: Press it.",
                    latest_speaker="Aggressive",
                    count=1,
                ),
            }
            update = await conservative_debator_node(state, {})

        debate = update["risk_debate"]
        assert debate["count"] == 2
        assert debate["latest_speaker"] == "Conservative"
        assert debate["current_conservative_response"].startswith(CONSERVATIVE_TAG)
        assert "vol is elevated" in debate["current_conservative_response"]
        # Aggressive history unchanged on a Conservative turn.
        assert debate["aggressive_history"] == "Aggressive Analyst: Press it."
        assert "Press it" in debate["history"]
        assert "vol is elevated" in debate["history"]


@pytest.mark.unit
@pytest.mark.asyncio
class TestNeutralDebatorNode:
    async def test_preserves_other_histories(self):
        agent = AsyncMock()
        agent.ainvoke.return_value = _ai(
            "Both sides over-press — scale in via a starter."
        )

        with patch(
            "muffin_agent.agents.trading_decision.nodes.create_neutral_debator_agent",
            AsyncMock(return_value=agent),
        ):
            state = {
                "analysis_context": _ctx(),
                "investment_judge": _judge(),
                "trader": _trader(),
                "risk_debate": _empty_risk_debate(
                    history="Aggressive Analyst: A\n\nConservative Analyst: C",
                    aggressive_history="Aggressive Analyst: A",
                    conservative_history="Conservative Analyst: C",
                    current_aggressive_response="Aggressive Analyst: A",
                    current_conservative_response="Conservative Analyst: C",
                    latest_speaker="Conservative",
                    count=2,
                ),
            }
            update = await neutral_debator_node(state, {})

        debate = update["risk_debate"]
        assert debate["count"] == 3
        assert debate["latest_speaker"] == "Neutral"
        assert debate["current_neutral_response"].startswith(NEUTRAL_TAG)
        # Both prior histories preserved.
        assert debate["aggressive_history"] == "Aggressive Analyst: A"
        assert debate["conservative_history"] == "Conservative Analyst: C"
        # Neutral's own history populated.
        assert "scale in" in debate["neutral_history"]


# ── Portfolio Manager node ───────────────────────────────────────────────────


@pytest.mark.unit
@pytest.mark.asyncio
class TestPortfolioManagerNode:
    async def test_happy_path_writes_structured_decision(self):
        agent = AsyncMock()
        agent.ainvoke.return_value = _pm_result(_pm_payload())

        with patch(
            "muffin_agent.agents.trading_decision.nodes.create_portfolio_manager_agent",
            AsyncMock(return_value=agent),
        ):
            state = {
                "analysis_context": _ctx(),
                "investment_judge": _judge(),
                "trader": _trader(),
                "risk_debate": _empty_risk_debate(
                    history="Aggressive: a\n\nConservative: c\n\nNeutral: n",
                    count=3,
                    latest_speaker="Neutral",
                ),
            }
            update = await portfolio_manager_node(state, {})

        decision = update["portfolio_decision"]
        assert decision["rating"] == "buy"
        assert decision["price_target"] == 220.0
        assert decision["confidence"] == 0.65
        assert "error" not in decision

        # Risk debate state updated with PM as latest_speaker.
        assert update["risk_debate"]["latest_speaker"] == "Portfolio Manager"
        assert update["risk_debate"]["judge_decision"] == "Buy AAPL at 2% NAV."

    async def test_skips_when_judge_or_trader_missing(self):
        factory = AsyncMock()
        with patch(
            "muffin_agent.agents.trading_decision.nodes.create_portfolio_manager_agent",
            factory,
        ):
            state = {
                "analysis_context": _ctx(),
                "investment_judge": _judge(),
                # trader missing
                "risk_debate": _empty_risk_debate(history="some", count=3),
            }
            update = await portfolio_manager_node(state, {})

        assert factory.await_count == 0
        assert update["portfolio_decision"]["rating"] == "hold"
        assert "Missing" in update["portfolio_decision"]["error"]

    async def test_skips_when_risk_debate_empty(self):
        factory = AsyncMock()
        with patch(
            "muffin_agent.agents.trading_decision.nodes.create_portfolio_manager_agent",
            factory,
        ):
            state = {
                "analysis_context": _ctx(),
                "investment_judge": _judge(),
                "trader": _trader(),
                # risk_debate has no history
            }
            update = await portfolio_manager_node(state, {})

        assert factory.await_count == 0
        assert "No risk debate transcript" in update["portfolio_decision"]["error"]

    async def test_agent_exception_yields_fallback(self):
        agent = AsyncMock()
        agent.ainvoke.side_effect = RuntimeError("PM failed")

        with patch(
            "muffin_agent.agents.trading_decision.nodes.create_portfolio_manager_agent",
            AsyncMock(return_value=agent),
        ):
            state = {
                "analysis_context": _ctx(),
                "investment_judge": _judge(),
                "trader": _trader(),
                "risk_debate": _empty_risk_debate(history="transcript", count=3),
            }
            update = await portfolio_manager_node(state, {})

        assert update["portfolio_decision"]["rating"] == "hold"
        assert "raised" in update["portfolio_decision"]["error"].lower()

    async def test_missing_structured_response_yields_fallback(self):
        agent = AsyncMock()
        agent.ainvoke.return_value = _ai("freeform pm output")

        with patch(
            "muffin_agent.agents.trading_decision.nodes.create_portfolio_manager_agent",
            AsyncMock(return_value=agent),
        ):
            state = {
                "analysis_context": _ctx(),
                "investment_judge": _judge(),
                "trader": _trader(),
                "risk_debate": _empty_risk_debate(history="transcript", count=3),
            }
            update = await portfolio_manager_node(state, {})

        assert update["portfolio_decision"]["rating"] == "hold"
        assert "raw_output" in update["portfolio_decision"]
        assert "freeform" in update["portfolio_decision"]["raw_output"]

    async def test_accepts_dict_structured_response(self):
        agent = AsyncMock()
        agent.ainvoke.return_value = {
            "messages": [AIMessage(content="...")],
            "structured_response": _pm_payload(rating="strong_buy"),
        }

        with patch(
            "muffin_agent.agents.trading_decision.nodes.create_portfolio_manager_agent",
            AsyncMock(return_value=agent),
        ):
            state = {
                "analysis_context": _ctx(),
                "investment_judge": _judge(),
                "trader": _trader(),
                "risk_debate": _empty_risk_debate(history="transcript", count=3),
            }
            update = await portfolio_manager_node(state, {})

        assert update["portfolio_decision"]["rating"] == "strong_buy"
