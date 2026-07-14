"""Tests for the Investment Judge node function.

The Bull/Bear researchers are no longer standalone nodes — they run inside
the ``investment_debate`` conference subgraph (covered by the conference
framework tests + the graph e2e). The Judge stays a plain parent-graph
node that synthesises the completed ``investment_debate_messages``
transcript.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage

from muffin_agent.agents.trading_decision.researchers import investment_judge
from muffin_agent.agents.trading_decision.researchers.investment_judge import (
    investment_judge_node,
)
from muffin_agent.agents.trading_decision.schemas import InvestmentJudgeOutput

from .conftest import fake_model_config


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

    def _debate(self) -> list[AIMessage]:
        return [
            AIMessage(content="B1", name="bull_researcher"),
            AIMessage(content="b1", name="bear_researcher"),
            AIMessage(content="B2", name="bull_researcher"),
            AIMessage(content="b2", name="bear_researcher"),
        ]

    async def test_writes_structured_judge_output(self):
        with patch.object(
            investment_judge.ModelConfiguration,
            "from_runnable_config",
            return_value=fake_model_config(self._judge_output()),
        ):
            state = {"investment_debate_messages": self._debate()[:2]}
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
            state = {"investment_debate_messages": self._debate()}
            await investment_judge_node(state, {})

        system_msg = captured.get_llm_for_role.return_value[0].invocations[0][0]
        assert "bull_researcher: B1" in system_msg.content
        assert "bear_researcher: b1" in system_msg.content
        assert "bull_researcher: B2" in system_msg.content
        assert "bear_researcher: b2" in system_msg.content
