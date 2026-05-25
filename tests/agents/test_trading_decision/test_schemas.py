"""Schema validation tests for the trading_decision module."""

import pytest
from pydantic import ValidationError

from muffin_agent.agents.trading_decision.schemas import InvestmentJudgeOutput


@pytest.mark.unit
class TestInvestmentJudgeOutput:
    def _valid_payload(self, **overrides):
        base = {
            "signal": "buy",
            "conviction": 0.7,
            "summary": "Bull thesis holds — earnings revisions improving.",
            "bull_case": "Services revenue growth + iPhone refresh cycle.",
            "bear_case": "China demand fragility and regulatory overhang.",
            "key_catalysts": ["Q1 earnings", "WWDC announcements"],
            "key_risks": ["China demand weakness"],
            "monitoring_checklist": [
                "Services growth rate",
                "iPhone unit shipments",
                "FX impact on EM revenue",
            ],
            "winning_side": "bull",
            "reasoning": "Bull addressed every bear point; bear conceded on services.",
        }
        base.update(overrides)
        return base

    def test_minimal_valid(self):
        out = InvestmentJudgeOutput(**self._valid_payload())
        assert out.signal == "buy"
        assert out.conviction == 0.7
        assert out.winning_side == "bull"

    def test_signal_must_be_in_enum(self):
        with pytest.raises(ValidationError):
            InvestmentJudgeOutput(**self._valid_payload(signal="overweight"))

    def test_winning_side_must_be_in_enum(self):
        with pytest.raises(ValidationError):
            InvestmentJudgeOutput(**self._valid_payload(winning_side="neither"))

    def test_conviction_lower_bound(self):
        with pytest.raises(ValidationError):
            InvestmentJudgeOutput(**self._valid_payload(conviction=-0.1))

    def test_conviction_upper_bound(self):
        with pytest.raises(ValidationError):
            InvestmentJudgeOutput(**self._valid_payload(conviction=1.1))

    def test_lists_default_to_empty(self):
        payload = self._valid_payload()
        del payload["key_catalysts"]
        del payload["key_risks"]
        del payload["monitoring_checklist"]
        out = InvestmentJudgeOutput(**payload)
        assert out.key_catalysts == []
        assert out.key_risks == []
        assert out.monitoring_checklist == []

    def test_round_trip_json(self):
        out = InvestmentJudgeOutput(**self._valid_payload())
        rehydrated = InvestmentJudgeOutput.model_validate(out.model_dump())
        assert rehydrated == out
