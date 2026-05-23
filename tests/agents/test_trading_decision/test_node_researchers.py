"""Tests for the Bull / Bear / Investment Judge node functions."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from muffin_agent.agents.trading_decision.researchers import (
    bear_researcher,
    bull_researcher,
    investment_judge,
)
from muffin_agent.agents.trading_decision.researchers.bear_researcher import (
    bear_researcher_node,
)
from muffin_agent.agents.trading_decision.researchers.bull_researcher import (
    bull_researcher_node,
)
from muffin_agent.agents.trading_decision.researchers.investment_judge import (
    investment_judge_node,
)
from muffin_agent.agents.trading_decision.schemas import InvestmentJudgeOutput

from .conftest import ai, fake_model_config


def _analysis_context() -> dict:
    return {
        "ticker": "AAPL",
        "query": "long-term hold",
        "narrative": "AAPL trades at 22x P/E.",
    }


@pytest.mark.unit
@pytest.mark.asyncio
class TestBullResearcherNode:
    async def test_opening_turn_writes_to_bull_responses(self):
        with patch.object(
            bull_researcher.ModelConfiguration,
            "from_runnable_config",
            return_value=fake_model_config(ai("bull opening argument")),
        ):
            state = {"analysis_context": _analysis_context()}
            update = await bull_researcher_node(state, {})

        assert update == {"investment_bull_responses": ["bull opening argument"]}

    async def test_reads_opposing_from_bear_responses(self):
        captured = fake_model_config(ai("rebuttal text"))
        with patch.object(
            bull_researcher.ModelConfiguration,
            "from_runnable_config",
            return_value=captured,
        ):
            state = {
                "analysis_context": _analysis_context(),
                "investment_bull_responses": ["my opener"],
                "investment_bear_responses": ["bear point 1", "bear point 2"],
            }
            await bull_researcher_node(state, {})

        # The fake LLM captured the messages; check the system prompt
        # included the latest Bear turn as opposing_last.
        fake_llm = captured.get_llm_for_role.return_value[0]
        system_msg_content = fake_llm.invocations[0][0].content
        assert "bear point 2" in system_msg_content
        # The full debate history should also include earlier turns.
        assert "Bull Researcher: my opener" in system_msg_content
        assert "Bear Researcher: bear point 1" in system_msg_content


@pytest.mark.unit
@pytest.mark.asyncio
class TestBearResearcherNode:
    async def test_opening_turn_writes_to_bear_responses(self):
        with patch.object(
            bear_researcher.ModelConfiguration,
            "from_runnable_config",
            return_value=fake_model_config(ai("bear opening")),
        ):
            state = {"analysis_context": _analysis_context()}
            update = await bear_researcher_node(state, {})

        assert update == {"investment_bear_responses": ["bear opening"]}

    async def test_reads_opposing_from_bull_responses(self):
        captured = fake_model_config(ai("rebut"))
        with patch.object(
            bear_researcher.ModelConfiguration,
            "from_runnable_config",
            return_value=captured,
        ):
            state = {
                "analysis_context": _analysis_context(),
                "investment_bull_responses": ["bull opener"],
            }
            await bear_researcher_node(state, {})

        system_msg_content = (
            captured.get_llm_for_role.return_value[0].invocations[0][0].content
        )
        assert "bull opener" in system_msg_content


@pytest.mark.unit
@pytest.mark.asyncio
class TestInvestmentJudgeNode:
    def _judge_output(self, **overrides) -> InvestmentJudgeOutput:
        base = {
            "signal": "buy",
            "conviction": 0.72,
            "summary": "Net bull.",
            "bull_case": "Services growth.",
            "bear_case": "China wobble.",
            "key_catalysts": ["Q1"],
            "key_risks": ["China"],
            "monitoring_checklist": ["Services growth"],
            "winning_side": "bull",
            "reasoning": "Bull addressed bear points.",
        }
        base.update(overrides)
        return InvestmentJudgeOutput.model_validate(base)

    async def test_writes_structured_judge_output(self):
        with patch.object(
            investment_judge.ModelConfiguration,
            "from_runnable_config",
            return_value=fake_model_config(self._judge_output()),
        ):
            state = {
                "analysis_context": _analysis_context(),
                "investment_bull_responses": ["bull arg"],
                "investment_bear_responses": ["bear arg"],
            }
            update = await investment_judge_node(state, {})

        assert "investment_judge" in update
        judge = update["investment_judge"]
        assert judge["signal"] == "buy"
        assert judge["conviction"] == 0.72
        assert judge["winning_side"] == "bull"

    async def test_prompt_includes_full_debate_history(self):
        captured = fake_model_config(self._judge_output())
        with patch.object(
            investment_judge.ModelConfiguration,
            "from_runnable_config",
            return_value=captured,
        ):
            state = {
                "analysis_context": _analysis_context(),
                "investment_bull_responses": ["B1", "B2"],
                "investment_bear_responses": ["b1", "b2"],
            }
            await investment_judge_node(state, {})

        system_msg = captured.get_llm_for_role.return_value[0].invocations[0][0]
        assert "Bull Researcher: B1" in system_msg.content
        assert "Bear Researcher: b1" in system_msg.content
        assert "Bull Researcher: B2" in system_msg.content
        assert "Bear Researcher: b2" in system_msg.content
