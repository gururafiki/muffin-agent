"""Jinja2 prompt rendering tests for the risk debate + Portfolio Manager (PR 3)."""

import pytest

from muffin_agent.prompts import render_template

AGGRESSIVE_TEMPLATE = "trading_decision/risk_debate/aggressive.jinja"
CONSERVATIVE_TEMPLATE = "trading_decision/risk_debate/conservative.jinja"
NEUTRAL_TEMPLATE = "trading_decision/risk_debate/neutral.jinja"
PM_TEMPLATE = "trading_decision/portfolio_manager.jinja"


def _judge_payload(**overrides) -> dict:
    base = {
        "signal": "buy",
        "conviction": 0.7,
        "key_catalysts": ["catalyst A"],
        "key_risks": ["risk A"],
    }
    base.update(overrides)
    return base


def _trader_payload(**overrides) -> dict:
    base = {
        "action": "buy",
        "entry_price": 195.0,
        "stop_loss": 180.0,
        "take_profit": 225.0,
        "position_sizing": "2% of NAV starter",
        "time_horizon": "3–6 months",
        "reasoning": "Judge conviction supports starter long.",
    }
    base.update(overrides)
    return base


def _debater_vars(**overrides):
    base = {
        "ticker": "AAPL",
        "query": None,
        "investment_judge": _judge_payload(),
        "trader": _trader_payload(),
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
class TestAggressivePrompt:
    def test_renders_basic(self):
        result = render_template(AGGRESSIVE_TEMPLATE, **_debater_vars())
        assert "Aggressive Risk Analyst" in result
        assert "AAPL" in result
        assert "opening Aggressive argument" in result

    def test_persona_pushes_back_on_caution(self):
        result = render_template(AGGRESSIVE_TEMPLATE, **_debater_vars()).lower()
        assert "push back" in result or "press" in result
        assert "rebut" in result

    def test_includes_judge_and_trader(self):
        result = render_template(AGGRESSIVE_TEMPLATE, **_debater_vars())
        assert '"signal": "buy"' in result
        assert '"entry_price": 195.0' in result

    def test_includes_opposing_argument(self):
        result = render_template(
            AGGRESSIVE_TEMPLATE,
            **_debater_vars(opposing_last="Conservative Analyst: stop is too wide."),
        )
        assert "Conservative Analyst: stop is too wide." in result
        assert "opening Aggressive argument" not in result


@pytest.mark.unit
class TestConservativePrompt:
    def test_renders_basic(self):
        result = render_template(CONSERVATIVE_TEMPLATE, **_debater_vars())
        assert "Conservative Risk Analyst" in result
        assert "opening Conservative argument" in result

    def test_focuses_on_downside(self):
        result = render_template(CONSERVATIVE_TEMPLATE, **_debater_vars()).lower()
        assert "downside" in result or "drawdown" in result
        assert "rebut" in result

    def test_includes_judge_and_trader(self):
        result = render_template(CONSERVATIVE_TEMPLATE, **_debater_vars())
        assert '"signal": "buy"' in result
        assert '"entry_price": 195.0' in result


@pytest.mark.unit
class TestNeutralPrompt:
    def test_renders_basic(self):
        result = render_template(NEUTRAL_TEMPLATE, **_debater_vars())
        assert "Neutral Risk Analyst" in result
        assert "opening Neutral argument" in result

    def test_critiques_both_extremes(self):
        result = render_template(NEUTRAL_TEMPLATE, **_debater_vars()).lower()
        assert "aggressive" in result
        assert "conservative" in result
        assert "balanced" in result


# ── Portfolio Manager prompt ─────────────────────────────────────────────────


def _pm_vars(**overrides):
    base = {
        "ticker": "AAPL",
        "query": None,
        "investment_judge": _judge_payload(),
        "trader": _trader_payload(),
        "risk_debate_history": (
            "Aggressive Analyst: ...\n\n"
            "Conservative Analyst: ...\n\n"
            "Neutral Analyst: ..."
        ),
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
class TestPortfolioManagerPrompt:
    def test_renders_basic(self):
        result = render_template(PM_TEMPLATE, **_pm_vars())
        assert "Portfolio Manager" in result
        assert "AAPL" in result
        assert "PortfolioDecisionOutput" in result

    def test_signal_scale_listed(self):
        result = render_template(PM_TEMPLATE, **_pm_vars())
        for tier in ("strong_buy", "buy", "hold", "sell", "strong_sell"):
            assert tier in result

    def test_hold_reservation_discipline_present(self):
        result = render_template(PM_TEMPLATE, **_pm_vars())
        assert "do not default to hold" in result.lower()

    def test_includes_judge_trader_and_risk_debate(self):
        result = render_template(PM_TEMPLATE, **_pm_vars())
        assert '"signal": "buy"' in result
        assert '"entry_price": 195.0' in result
        assert "Aggressive Analyst" in result

    def test_pr4_reflection_flag_referenced(self):
        # Ensures the prompt doesn't accidentally claim past lessons before PR 4.
        result = render_template(PM_TEMPLATE, **_pm_vars())
        assert "incorporates_past_lessons" in result
