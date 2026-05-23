"""Tests for the 3 risk-debate node functions."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from muffin_agent.agents.trading_decision.risk_debate import (
    aggressive_debator,
    conservative_debator,
    neutral_debator,
)
from muffin_agent.agents.trading_decision.risk_debate.aggressive_debator import (
    aggressive_debator_node,
)
from muffin_agent.agents.trading_decision.risk_debate.conservative_debator import (
    conservative_debator_node,
)
from muffin_agent.agents.trading_decision.risk_debate.neutral_debator import (
    neutral_debator_node,
)

from .conftest import ai, fake_model_config


def _base_state() -> dict:
    return {
        "analysis_context": {"ticker": "AAPL"},
        "investment_judge": {"signal": "buy", "conviction": 0.7},
        "trader": {"action": "buy", "position_sizing": "2% NAV"},
    }


@pytest.mark.unit
@pytest.mark.asyncio
class TestAggressiveDebatorNode:
    async def test_writes_to_aggressive_responses(self):
        with patch.object(
            aggressive_debator.ModelConfiguration,
            "from_runnable_config",
            return_value=fake_model_config(ai("press harder")),
        ):
            update = await aggressive_debator_node(_base_state(), {})

        assert update == {"risk_aggressive_responses": ["press harder"]}

    async def test_includes_prior_debate_in_prompt(self):
        captured = fake_model_config(ai("aggressive rebut"))
        with patch.object(
            aggressive_debator.ModelConfiguration,
            "from_runnable_config",
            return_value=captured,
        ):
            state = _base_state()
            state["risk_conservative_responses"] = ["be careful"]
            state["risk_neutral_responses"] = ["scale in"]
            await aggressive_debator_node(state, {})

        system_msg = captured.get_llm_for_role.return_value[0].invocations[0][0]
        assert "Conservative Analyst: be careful" in system_msg.content
        assert "Neutral Analyst: scale in" in system_msg.content


@pytest.mark.unit
@pytest.mark.asyncio
class TestConservativeDebatorNode:
    async def test_writes_to_conservative_responses(self):
        with patch.object(
            conservative_debator.ModelConfiguration,
            "from_runnable_config",
            return_value=fake_model_config(ai("tighten stops")),
        ):
            update = await conservative_debator_node(_base_state(), {})

        assert update == {"risk_conservative_responses": ["tighten stops"]}


@pytest.mark.unit
@pytest.mark.asyncio
class TestNeutralDebatorNode:
    async def test_writes_to_neutral_responses(self):
        with patch.object(
            neutral_debator.ModelConfiguration,
            "from_runnable_config",
            return_value=fake_model_config(ai("balanced approach")),
        ):
            update = await neutral_debator_node(_base_state(), {})

        assert update == {"risk_neutral_responses": ["balanced approach"]}
