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
class TestFromInvestmentAnalysisState:
    """Adapter: ``TickerAnalysisState`` dict → ``AnalysisContext`` (PR 5)."""

    def _full_state(self) -> dict:
        return {
            "ticker": "AAPL",
            "query": "long-term hold",
            "market_regime": {"regime_label": "expansion", "vix": 14.2},
            "sector_view": {"sector": "tech", "valuation_signal": "fair"},
            "company_analysis": {"company_signal": "pass", "moat_width": "wide"},
            "forecast": {"revenue_growth_3y_cagr": 0.08},
            "risk_assessment": {"var_95_1m_pct": 6.4},
            "valuation": {"valuation_signal": "cheap"},
            # Non-context keys should be ignored.
            "thesis": {"signal": "hold"},
            "messages": [],
        }

    def test_full_state_maps_all_six_sections(self):
        ctx = AnalysisContext.from_investment_analysis_state(self._full_state())
        assert ctx.ticker == "AAPL"
        assert ctx.query == "long-term hold"
        assert ctx.market_regime == {"regime_label": "expansion", "vix": 14.2}
        assert ctx.sector_view == {"sector": "tech", "valuation_signal": "fair"}
        assert ctx.company_analysis == {
            "company_signal": "pass",
            "moat_width": "wide",
        }
        assert ctx.forecast == {"revenue_growth_3y_cagr": 0.08}
        assert ctx.risk_assessment == {"var_95_1m_pct": 6.4}
        assert ctx.valuation == {"valuation_signal": "cheap"}
        # Non-context keys NOT promoted into AnalysisContext.
        assert ctx.narrative is None
        assert ctx.additional_context == {}

    def test_partial_state_drops_missing_sections_to_none(self):
        state = {
            "ticker": "MSFT",
            "market_regime": {"regime_label": "neutral"},
            "company_analysis": {"company_signal": "watch"},
            # No sector_view / forecast / risk_assessment / valuation.
        }
        ctx = AnalysisContext.from_investment_analysis_state(state)
        assert ctx.ticker == "MSFT"
        assert ctx.market_regime is not None
        assert ctx.sector_view is None
        assert ctx.forecast is None
        assert ctx.risk_assessment is None
        assert ctx.valuation is None
        assert ctx.company_analysis is not None

    def test_empty_dict_sections_treated_as_missing(self):
        """An empty dict from a failed/skipped node should map to None, not {}."""
        state = {
            "ticker": "AAPL",
            "market_regime": {},  # node returned empty
            "valuation": {"valuation_signal": "cheap"},
        }
        ctx = AnalysisContext.from_investment_analysis_state(state)
        assert ctx.market_regime is None
        assert ctx.valuation == {"valuation_signal": "cheap"}

    def test_non_dict_section_treated_as_missing(self):
        """A non-dict section (e.g. error string) should map to None."""
        state = {
            "ticker": "AAPL",
            "market_regime": "error string",  # malformed upstream
            "valuation": ["unexpected", "shape"],
        }
        ctx = AnalysisContext.from_investment_analysis_state(state)
        assert ctx.market_regime is None
        assert ctx.valuation is None

    def test_ticker_override(self):
        state = {"ticker": "AAPL", "market_regime": {"r": "x"}}
        ctx = AnalysisContext.from_investment_analysis_state(state, ticker="GOOG")
        assert ctx.ticker == "GOOG"

    def test_query_override(self):
        state = {"ticker": "AAPL", "query": "original mandate"}
        ctx = AnalysisContext.from_investment_analysis_state(
            state, query="overridden mandate"
        )
        assert ctx.query == "overridden mandate"

    def test_narrative_kwarg_supplements_structured_fields(self):
        ctx = AnalysisContext.from_investment_analysis_state(
            self._full_state(),
            narrative="Add'l notes: management transition imminent.",
        )
        assert ctx.market_regime is not None
        assert ctx.narrative == "Add'l notes: management transition imminent."

    def test_missing_ticker_raises(self):
        with pytest.raises(ValueError, match="non-empty ticker"):
            AnalysisContext.from_investment_analysis_state({"market_regime": {}})

    def test_empty_ticker_in_state_with_override_succeeds(self):
        # Override path lets callers recover from upstream-missing ticker.
        ctx = AnalysisContext.from_investment_analysis_state(
            {"ticker": "", "market_regime": {"r": "x"}}, ticker="AAPL"
        )
        assert ctx.ticker == "AAPL"

    def test_empty_state_with_just_ticker_kwarg(self):
        """Adapter works when only a ticker is provided — minimal valid context."""
        ctx = AnalysisContext.from_investment_analysis_state({}, ticker="AAPL")
        assert ctx.ticker == "AAPL"
        assert ctx.market_regime is None

    def test_round_trip_via_json_dump(self):
        """Pipe-friendly: serialise a state, parse JSON, run adapter."""
        import json

        state_json = json.dumps(self._full_state())
        rehydrated = json.loads(state_json)
        ctx = AnalysisContext.from_investment_analysis_state(rehydrated)
        assert ctx.ticker == "AAPL"
        assert ctx.market_regime == {"regime_label": "expansion", "vix": 14.2}


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
