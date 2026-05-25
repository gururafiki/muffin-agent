"""Tests for ``TradingDecisionConfiguration``."""

from __future__ import annotations

import pytest
from langchain_core.runnables import RunnableConfig
from pydantic import ValidationError

from muffin_agent.agents.trading_decision.config import (
    TradingDecisionConfiguration,
)


@pytest.mark.unit
class TestTradingDecisionConfiguration:
    def test_defaults(self):
        cfg = TradingDecisionConfiguration()
        assert cfg.max_investment_debate_rounds == 2
        assert cfg.max_risk_debate_rounds == 1
        assert cfg.reflection_enabled is True
        assert cfg.reflection_holding_days == 5
        assert cfg.reflection_benchmark == "SPY"
        assert cfg.reflection_max_same_ticker == 5
        assert cfg.reflection_max_cross_ticker == 3
        assert cfg.decision_date is None

    def test_from_runnable_config_pulls_configurable(self):
        rc: RunnableConfig = {
            "configurable": {
                "max_investment_debate_rounds": 3,
                "max_risk_debate_rounds": 2,
                "reflection_enabled": False,
                "reflection_benchmark": "QQQ",
                "decision_date": "2026-05-17",
            }
        }
        cfg = TradingDecisionConfiguration.from_runnable_config(rc)
        assert cfg.max_investment_debate_rounds == 3
        assert cfg.max_risk_debate_rounds == 2
        assert cfg.reflection_enabled is False
        assert cfg.reflection_benchmark == "QQQ"
        assert cfg.decision_date == "2026-05-17"

    def test_unknown_keys_ignored(self):
        rc: RunnableConfig = {
            "configurable": {
                "unknown_key": "value",
                "max_investment_debate_rounds": 4,
            }
        }
        cfg = TradingDecisionConfiguration.from_runnable_config(rc)
        assert cfg.max_investment_debate_rounds == 4

    def test_rounds_must_be_positive(self):
        with pytest.raises(ValidationError):
            TradingDecisionConfiguration(max_investment_debate_rounds=0)
        with pytest.raises(ValidationError):
            TradingDecisionConfiguration(max_risk_debate_rounds=0)
