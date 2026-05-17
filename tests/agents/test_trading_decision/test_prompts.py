"""Jinja2 prompt template rendering tests for trading_decision."""

import pytest

from muffin_agent.prompts import render_template

BULL_TEMPLATE = "trading_decision/researchers/bull.jinja"
BEAR_TEMPLATE = "trading_decision/researchers/bear.jinja"
JUDGE_TEMPLATE = "trading_decision/researchers/investment_judge.jinja"


def _bare_vars(**overrides):
    """Minimum prompt vars required for any researcher prompt."""
    base = {
        "ticker": "AAPL",
        "query": None,
        "debate_history": "",
        "opposing_last": "",
        "market_regime": None,
        "sector_view": None,
        "company_analysis": None,
        "forecast": None,
        "risk_assessment": None,
        "valuation": None,
        "narrative": None,
        "additional_context": {},
    }
    base.update(overrides)
    return base


@pytest.mark.unit
class TestBullPrompt:
    def test_renders_without_context(self):
        result = render_template(BULL_TEMPLATE, **_bare_vars())
        assert "Bull Researcher" in result
        assert "AAPL" in result
        assert "opening argument" in result.lower()

    def test_renders_with_narrative(self):
        result = render_template(
            BULL_TEMPLATE,
            **_bare_vars(narrative="Apple has strong AI roadmap."),
        )
        assert "Apple has strong AI roadmap." in result

    def test_renders_with_structured_context(self):
        result = render_template(
            BULL_TEMPLATE,
            **_bare_vars(
                valuation={"signal": "cheap", "dcf_per_share": 220.0},
                forecast={"revenue_growth_3y_cagr": 0.08},
            ),
        )
        assert '"signal": "cheap"' in result
        assert "0.08" in result

    def test_engagement_rules_present(self):
        result = render_template(BULL_TEMPLATE, **_bare_vars())
        assert "engage directly" in result.lower()
        assert "rebut" in result.lower()
        assert "may not invent" in result.lower() or "may NOT invent" in result

    def test_includes_opposing_argument_when_present(self):
        result = render_template(
            BULL_TEMPLATE,
            **_bare_vars(opposing_last="Bear Researcher: services growth is mature."),
        )
        assert "Bear Researcher: services growth is mature." in result
        assert "opening argument" not in result.lower()


@pytest.mark.unit
class TestBearPrompt:
    def test_renders_without_context(self):
        result = render_template(BEAR_TEMPLATE, **_bare_vars())
        assert "Bear Researcher" in result
        assert "AAPL" in result
        assert "opening bear argument" in result.lower()

    def test_engagement_rules_present(self):
        result = render_template(BEAR_TEMPLATE, **_bare_vars())
        assert "engage directly" in result.lower()
        assert "rebut" in result.lower()

    def test_includes_opposing_argument_when_present(self):
        result = render_template(
            BEAR_TEMPLATE,
            **_bare_vars(opposing_last="Bull Researcher: AI cycle is strong."),
        )
        assert "Bull Researcher: AI cycle is strong." in result


@pytest.mark.unit
class TestJudgePrompt:
    def _judge_vars(self, **overrides):
        base = {
            "ticker": "AAPL",
            "query": None,
            "debate_history": "Bull Researcher: ...\n\nBear Researcher: ...",
            "market_regime": None,
            "sector_view": None,
            "company_analysis": None,
            "forecast": None,
            "risk_assessment": None,
            "valuation": None,
            "narrative": None,
            "additional_context": {},
        }
        base.update(overrides)
        return base

    def test_renders_basic(self):
        result = render_template(JUDGE_TEMPLATE, **self._judge_vars())
        assert "Investment Judge" in result
        assert "AAPL" in result
        assert "Bull Researcher: ..." in result
        assert "Bear Researcher: ..." in result

    def test_signal_scale_listed(self):
        result = render_template(JUDGE_TEMPLATE, **self._judge_vars())
        for tier in ("strong_buy", "buy", "hold", "sell", "strong_sell"):
            assert tier in result

    def test_hold_reservation_instruction_present(self):
        result = render_template(JUDGE_TEMPLATE, **self._judge_vars())
        # Critical prompt discipline borrowed from TradingAgents — prevents
        # default-to-hold behaviour under any uncertainty.
        assert "do not default to hold" in result.lower()

    def test_renders_with_structured_context(self):
        result = render_template(
            JUDGE_TEMPLATE,
            **self._judge_vars(
                valuation={"signal": "expensive"},
                risk_assessment={"var_95_1m_pct": 8.4},
            ),
        )
        assert "expensive" in result
        assert "8.4" in result

    def test_output_schema_referenced(self):
        result = render_template(JUDGE_TEMPLATE, **self._judge_vars())
        assert "InvestmentJudgeOutput" in result
