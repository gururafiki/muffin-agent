"""Schema validation tests for the trading_decision module."""

import pytest
from pydantic import ValidationError

from muffin_agent.agents.trading_decision.schemas import (
    AnalysisContext,
    InvestmentJudgeOutput,
)


@pytest.mark.unit
class TestAnalysisContext:
    def test_minimal_valid(self):
        ctx = AnalysisContext(ticker="AAPL")
        assert ctx.ticker == "AAPL"
        assert ctx.query is None
        assert ctx.narrative is None
        assert ctx.additional_context == {}
        # All structured fields default to None
        assert ctx.market_regime is None
        assert ctx.valuation is None

    def test_from_narrative_factory(self):
        ctx = AnalysisContext.from_narrative(
            "AAPL",
            "Apple is well-positioned for AI device demand.",
        )
        assert ctx.ticker == "AAPL"
        assert ctx.narrative == "Apple is well-positioned for AI device demand."
        assert ctx.query is None
        assert ctx.market_regime is None

    def test_from_narrative_with_query(self):
        ctx = AnalysisContext.from_narrative(
            "AAPL",
            "Notes.",
            query="long-term hold",
        )
        assert ctx.query == "long-term hold"
        assert ctx.additional_context == {}

    def test_from_narrative_extra_kwargs_land_in_additional_context(self):
        ctx = AnalysisContext.from_narrative(
            "AAPL", "Notes.", portfolio_weight=0.05, horizon_months=12
        )
        assert ctx.additional_context == {
            "portfolio_weight": 0.05,
            "horizon_months": 12,
        }

    def test_structured_fields_accept_dicts(self):
        ctx = AnalysisContext(
            ticker="AAPL",
            market_regime={"regime_label": "expansion", "vix": 14.2},
            valuation={"signal": "fairly_valued"},
        )
        assert ctx.market_regime is not None
        assert ctx.market_regime["regime_label"] == "expansion"
        assert ctx.valuation is not None
        assert ctx.valuation["signal"] == "fairly_valued"

    def test_round_trip_json(self):
        ctx = AnalysisContext(
            ticker="JPM",
            query="value scan",
            narrative="JPM trades at 1.6x book.",
            market_regime={"regime_label": "neutral"},
        )
        rehydrated = AnalysisContext.model_validate(ctx.model_dump())
        assert rehydrated == ctx


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
