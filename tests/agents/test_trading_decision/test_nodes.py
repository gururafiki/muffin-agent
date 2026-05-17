"""Tests for the trading_decision node wrappers.

Patches the agent factories so the nodes execute their state-translation
logic against fully-controlled fake agents. Verifies:

* Bull / Bear nodes prepend the correct speaker tag.
* History strings accumulate correctly across turns.
* ``count`` increments by 1 per turn.
* ``current_response`` always reflects the latest speaker.
* Judge node extracts ``structured_response`` and dumps it.
* Judge node returns the fallback dict on missing structured output.
* All nodes accept ``analysis_context`` as either a Pydantic instance or a
  plain dict (for graph-level invocation).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.messages import AIMessage

from muffin_agent.agents.trading_decision.conditional_logic import (
    BEAR_TAG,
    BULL_TAG,
)
from muffin_agent.agents.trading_decision.nodes import (
    bear_researcher_node,
    bull_researcher_node,
    investment_judge_node,
)
from muffin_agent.agents.trading_decision.schemas import (
    AnalysisContext,
    InvestmentJudgeOutput,
)


def _ai_result(text: str) -> dict:
    """Shape mimicking a ReAct agent result with a single AIMessage reply."""
    return {"messages": [AIMessage(content=text)]}


def _judge_result(payload: dict) -> dict:
    """Shape mimicking a ReAct agent with structured output."""
    return {
        "messages": [AIMessage(content="(structured response delivered)")],
        "structured_response": InvestmentJudgeOutput.model_validate(payload),
    }


def _judge_payload(**overrides) -> dict:
    base = {
        "signal": "buy",
        "conviction": 0.72,
        "summary": "Net bull on services growth and undemanding valuation.",
        "bull_case": "Services revenue durable; iPhone refresh imminent.",
        "bear_case": "China demand wobble + ad-tech regulatory drag.",
        "key_catalysts": ["Q1 earnings", "WWDC announcements"],
        "key_risks": ["China demand"],
        "monitoring_checklist": ["Services growth", "iPhone units", "FX"],
        "winning_side": "bull",
        "reasoning": "Bull addressed every credible bear objection.",
    }
    base.update(overrides)
    return base


def _ctx(**overrides) -> AnalysisContext:
    overrides.setdefault("ticker", "AAPL")
    overrides.setdefault("narrative", "AAPL trades at 22x P/E.")
    return AnalysisContext(**overrides)


@pytest.mark.unit
@pytest.mark.asyncio
class TestBullResearcherNode:
    async def test_opening_turn_tags_and_increments(self):
        agent = AsyncMock()
        agent.ainvoke.return_value = _ai_result("AAPL has structural tailwinds.")

        with patch(
            "muffin_agent.agents.trading_decision.nodes.create_bull_researcher_agent",
            AsyncMock(return_value=agent),
        ):
            state: dict = {"analysis_context": _ctx()}
            update = await bull_researcher_node(state, {})

        debate = update["investment_debate"]
        assert debate["current_response"].startswith(BULL_TAG)
        assert "AAPL has structural tailwinds." in debate["current_response"]
        assert debate["count"] == 1
        assert debate["bull_history"].endswith("AAPL has structural tailwinds.")
        assert debate["bear_history"] == ""
        assert debate["history"].startswith(BULL_TAG)

    async def test_rebuttal_turn_preserves_prior_history(self):
        agent = AsyncMock()
        agent.ainvoke.return_value = _ai_result("Rebut: services are accelerating.")

        with patch(
            "muffin_agent.agents.trading_decision.nodes.create_bull_researcher_agent",
            AsyncMock(return_value=agent),
        ):
            state = {
                "analysis_context": _ctx(),
                "investment_debate": {
                    "history": "Bull Researcher: round1\n\nBear Researcher: round1",
                    "bull_history": "Bull Researcher: round1",
                    "bear_history": "Bear Researcher: round1",
                    "current_response": "Bear Researcher: round1",
                    "judge_decision": "",
                    "count": 2,
                },
            }
            update = await bull_researcher_node(state, {})

        debate = update["investment_debate"]
        assert debate["count"] == 3
        assert "round1" in debate["history"]
        assert "Rebut: services are accelerating." in debate["history"]
        assert "Rebut: services are accelerating." in debate["bull_history"]
        # Bear history untouched on a Bull turn
        assert debate["bear_history"] == "Bear Researcher: round1"

    async def test_accepts_dict_context(self):
        """Graph-level invocation passes analysis_context as a dict, not Pydantic."""
        agent = AsyncMock()
        agent.ainvoke.return_value = _ai_result("Bull says X.")

        with patch(
            "muffin_agent.agents.trading_decision.nodes.create_bull_researcher_agent",
            AsyncMock(return_value=agent),
        ):
            state = {"analysis_context": {"ticker": "AAPL", "narrative": "Notes."}}
            update = await bull_researcher_node(state, {})

        assert update["investment_debate"]["count"] == 1

    async def test_agent_exception_yields_failure_marker(self):
        agent = AsyncMock()
        agent.ainvoke.side_effect = RuntimeError("provider failure")

        with patch(
            "muffin_agent.agents.trading_decision.nodes.create_bull_researcher_agent",
            AsyncMock(return_value=agent),
        ):
            state = {"analysis_context": _ctx()}
            update = await bull_researcher_node(state, {})

        debate = update["investment_debate"]
        assert debate["count"] == 1
        assert debate["current_response"].startswith(BULL_TAG)
        assert "failed" in debate["current_response"].lower()


@pytest.mark.unit
@pytest.mark.asyncio
class TestBearResearcherNode:
    async def test_opening_turn_tags_and_increments(self):
        agent = AsyncMock()
        agent.ainvoke.return_value = _ai_result("China demand softness is real.")

        with patch(
            "muffin_agent.agents.trading_decision.nodes.create_bear_researcher_agent",
            AsyncMock(return_value=agent),
        ):
            state = {"analysis_context": _ctx()}
            update = await bear_researcher_node(state, {})

        debate = update["investment_debate"]
        assert debate["current_response"].startswith(BEAR_TAG)
        assert "China demand softness is real." in debate["current_response"]
        assert debate["count"] == 1
        assert debate["bear_history"].endswith("China demand softness is real.")
        assert debate["bull_history"] == ""

    async def test_rebuttal_preserves_bull_history(self):
        agent = AsyncMock()
        agent.ainvoke.return_value = _ai_result("Bear rebuts: services maturing.")

        with patch(
            "muffin_agent.agents.trading_decision.nodes.create_bear_researcher_agent",
            AsyncMock(return_value=agent),
        ):
            state = {
                "analysis_context": _ctx(),
                "investment_debate": {
                    "history": "Bull Researcher: round1",
                    "bull_history": "Bull Researcher: round1",
                    "bear_history": "",
                    "current_response": "Bull Researcher: round1",
                    "judge_decision": "",
                    "count": 1,
                },
            }
            update = await bear_researcher_node(state, {})

        debate = update["investment_debate"]
        assert debate["count"] == 2
        # Bull history unchanged on a Bear turn
        assert debate["bull_history"] == "Bull Researcher: round1"
        assert "Bear rebuts" in debate["bear_history"]


@pytest.mark.unit
@pytest.mark.asyncio
class TestInvestmentJudgeNode:
    async def test_synthesises_from_debate_history(self):
        agent = AsyncMock()
        agent.ainvoke.return_value = _judge_result(_judge_payload())

        with patch(
            "muffin_agent.agents.trading_decision.nodes.create_investment_judge_agent",
            AsyncMock(return_value=agent),
        ):
            state = {
                "analysis_context": _ctx(),
                "investment_debate": {
                    "history": "Bull Researcher: ...\n\nBear Researcher: ...",
                    "bull_history": "Bull Researcher: ...",
                    "bear_history": "Bear Researcher: ...",
                    "current_response": "Bear Researcher: ...",
                    "judge_decision": "",
                    "count": 2,
                },
            }
            update = await investment_judge_node(state, {})

        judge = update["investment_judge"]
        assert judge["signal"] == "buy"
        assert judge["winning_side"] == "bull"
        assert judge["conviction"] == 0.72
        assert "investment_debate" in update
        assert update["investment_debate"]["judge_decision"]

    async def test_empty_debate_history_short_circuits(self):
        """Judge should not invoke the LLM if there is no transcript to read."""
        factory_mock = AsyncMock()
        with patch(
            "muffin_agent.agents.trading_decision.nodes.create_investment_judge_agent",
            factory_mock,
        ):
            state = {"analysis_context": _ctx()}
            update = await investment_judge_node(state, {})

        assert factory_mock.await_count == 0
        assert update["investment_judge"]["signal"] == "hold"
        assert "no debate history" in update["investment_judge"]["error"].lower()

    async def test_missing_structured_response_yields_fallback(self):
        agent = AsyncMock()
        agent.ainvoke.return_value = _ai_result("freeform reply, no schema")

        with patch(
            "muffin_agent.agents.trading_decision.nodes.create_investment_judge_agent",
            AsyncMock(return_value=agent),
        ):
            state = {
                "analysis_context": _ctx(),
                "investment_debate": {
                    "history": "some turns",
                    "bull_history": "",
                    "bear_history": "",
                    "current_response": "",
                    "judge_decision": "",
                    "count": 2,
                },
            }
            update = await investment_judge_node(state, {})

        judge = update["investment_judge"]
        assert judge["signal"] == "hold"
        assert "raw_output" in judge
        assert judge["raw_output"] == "freeform reply, no schema"

    async def test_agent_exception_yields_fallback(self):
        agent = AsyncMock()
        agent.ainvoke.side_effect = RuntimeError("model exhausted")

        with patch(
            "muffin_agent.agents.trading_decision.nodes.create_investment_judge_agent",
            AsyncMock(return_value=agent),
        ):
            state = {
                "analysis_context": _ctx(),
                "investment_debate": {
                    "history": "some turns",
                    "bull_history": "",
                    "bear_history": "",
                    "current_response": "",
                    "judge_decision": "",
                    "count": 2,
                },
            }
            update = await investment_judge_node(state, {})

        assert update["investment_judge"]["signal"] == "hold"
        assert "raised" in update["investment_judge"]["error"].lower()

    async def test_accepts_dict_structured_response(self):
        """``response_format`` may return a dict instead of a Pydantic instance."""
        agent = AsyncMock()
        payload = _judge_payload(signal="strong_sell", winning_side="bear")
        agent.ainvoke.return_value = {
            "messages": [AIMessage(content="...")],
            "structured_response": payload,  # dict, not Pydantic
        }

        with patch(
            "muffin_agent.agents.trading_decision.nodes.create_investment_judge_agent",
            AsyncMock(return_value=agent),
        ):
            state = {
                "analysis_context": _ctx(),
                "investment_debate": {
                    "history": "transcript",
                    "bull_history": "",
                    "bear_history": "",
                    "current_response": "",
                    "judge_decision": "",
                    "count": 2,
                },
            }
            update = await investment_judge_node(state, {})

        assert update["investment_judge"]["signal"] == "strong_sell"
        assert update["investment_judge"]["winning_side"] == "bear"
