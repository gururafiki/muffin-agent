"""Tests for the Portfolio Manager node."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from muffin_agent.agents.trading_decision import portfolio_manager as pm_module
from muffin_agent.agents.trading_decision.portfolio_manager import (
    portfolio_manager_node,
)
from muffin_agent.agents.trading_decision.schemas import PortfolioDecisionOutput

from .conftest import fake_model_config


def _pm_output(**overrides) -> PortfolioDecisionOutput:
    base = {
        "rating": "buy",
        "executive_summary": "Buy AAPL at 2% NAV.",
        "investment_thesis": "Bull held after debate.",
        "price_target": 220.0,
        "stop_loss": 180.0,
        "time_horizon": "3-6 months",
        "position_sizing": "2% NAV starter",
        "key_risks_remaining": ["China demand"],
        "confidence": 0.65,
    }
    base.update(overrides)
    return PortfolioDecisionOutput.model_validate(base)


def _base_state() -> dict:
    return {
        "analysis_context": {"ticker": "AAPL"},
        "investment_judge": {"signal": "buy", "conviction": 0.7},
        "trader": {"action": "buy", "position_sizing": "2% NAV"},
        "risk_aggressive_responses": ["press it"],
        "risk_conservative_responses": ["careful"],
        "risk_neutral_responses": ["scale in"],
    }


@pytest.mark.unit
@pytest.mark.asyncio
class TestPortfolioManagerNode:
    async def test_writes_structured_decision(self):
        with patch.object(
            pm_module.ModelConfiguration,
            "from_runnable_config",
            return_value=fake_model_config(_pm_output()),
        ):
            update = await portfolio_manager_node(_base_state(), {})

        assert "portfolio_decision" in update
        decision = update["portfolio_decision"]
        assert decision["rating"] == "buy"
        assert decision["price_target"] == 220.0
        assert decision["incorporates_past_lessons"] is False

    async def test_prompt_includes_risk_debate_and_past_reflections(self):
        captured = fake_model_config(_pm_output())
        with patch.object(
            pm_module.ModelConfiguration,
            "from_runnable_config",
            return_value=captured,
        ):
            state = _base_state()
            state["past_reflections"] = "- **AAPL 2026-01-01** → buy | raw +5%"
            await portfolio_manager_node(state, {})

        system_msg = captured.get_llm_for_role.return_value[0].invocations[0][0]
        assert "Aggressive Analyst: press it" in system_msg.content
        assert "Conservative Analyst: careful" in system_msg.content
        assert "Neutral Analyst: scale in" in system_msg.content
        assert "Past lessons" in system_msg.content
        assert "AAPL 2026-01-01" in system_msg.content

    async def test_prompt_omits_past_reflections_when_empty(self):
        captured = fake_model_config(_pm_output())
        with patch.object(
            pm_module.ModelConfiguration,
            "from_runnable_config",
            return_value=captured,
        ):
            await portfolio_manager_node(_base_state(), {})

        system_msg = captured.get_llm_for_role.return_value[0].invocations[0][0]
        assert "Past lessons" not in system_msg.content
