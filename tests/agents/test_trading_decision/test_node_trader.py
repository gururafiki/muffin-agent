"""Tests for the Trader node."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from muffin_agent.agents.trading_decision import trader as trader_module
from muffin_agent.agents.trading_decision.schemas import TraderOutput
from muffin_agent.agents.trading_decision.trader import trader_node

from .conftest import fake_model_config


def _judge_payload() -> dict:
    return {
        "signal": "buy",
        "conviction": 0.7,
        "summary": "Net bull.",
        "bull_case": "Services growth.",
        "bear_case": "China wobble.",
        "key_catalysts": ["Q1"],
        "key_risks": ["China"],
        "monitoring_checklist": ["Services"],
        "winning_side": "bull",
        "reasoning": "Bull held.",
    }


def _trader_output(**overrides) -> TraderOutput:
    base = {
        "action": "buy",
        "reasoning": "Starter long.",
        "entry_price": 195.0,
        "stop_loss": 180.0,
        "take_profit": 225.0,
        "position_sizing": "2% NAV starter",
        "time_horizon": "3-6 months",
    }
    base.update(overrides)
    return TraderOutput.model_validate(base)


@pytest.mark.unit
@pytest.mark.asyncio
class TestTraderNode:
    async def test_writes_structured_trader_output(self):
        with patch.object(
            trader_module.ModelConfiguration,
            "from_runnable_config",
            return_value=fake_model_config(_trader_output()),
        ):
            state = {
                "analysis_context": {"ticker": "AAPL"},
                "investment_judge": _judge_payload(),
            }
            update = await trader_node(state, {})

        assert "trader" in update
        trader = update["trader"]
        assert trader["action"] == "buy"
        assert trader["entry_price"] == 195.0

    async def test_prompt_includes_judge_payload(self):
        captured = fake_model_config(_trader_output())
        with patch.object(
            trader_module.ModelConfiguration,
            "from_runnable_config",
            return_value=captured,
        ):
            state = {
                "analysis_context": {"ticker": "AAPL"},
                "investment_judge": _judge_payload(),
            }
            await trader_node(state, {})

        system_msg = captured.get_llm_for_role.return_value[0].invocations[0][0]
        # Judge payload appears as JSON in the prompt.
        assert "Net bull." in system_msg.content
        assert '"signal": "buy"' in system_msg.content
