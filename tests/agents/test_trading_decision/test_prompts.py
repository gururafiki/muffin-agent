"""Prompt-template rendering tests for trading_decision Jinja files.

Verifies that:
* Each per-role template renders without errors with the standard set of
  per-call vars.
* The shared partials (`_analysis_context.jinja`, `_investment_debate_state.jinja`,
  `_risk_synthesis_inputs.jinja`) are pulled in correctly via ``{% include %}``.
* Hold-reservation discipline appears in the synthesis prompts.
"""

from __future__ import annotations

from typing import Any

import pytest

from muffin_agent.prompts import render_template

# ── Common kwargs ─────────────────────────────────────────────────────────────


def _analysis_kwargs(**overrides) -> dict[str, Any]:
    base = {
        "ticker": "AAPL",
        "query": "long-term hold",
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


def _judge_payload() -> dict[str, Any]:
    return {
        "signal": "buy",
        "conviction": 0.7,
        "summary": "Net bull.",
        "bull_case": "Services growth durable.",
        "bear_case": "China demand wobble.",
        "key_catalysts": ["Q1 earnings"],
        "key_risks": ["China"],
        "monitoring_checklist": ["Services growth"],
        "winning_side": "bull",
        "reasoning": "Bull held.",
    }


def _trader_payload() -> dict[str, Any]:
    return {
        "action": "buy",
        "reasoning": "Starter long.",
        "entry_price": 195.0,
        "stop_loss": 180.0,
        "take_profit": 225.0,
        "position_sizing": "2% NAV",
        "time_horizon": "3-6m",
    }


# ── Investment debate templates ──────────────────────────────────────────────


@pytest.mark.unit
class TestInvestmentDebatePrompts:
    def test_bull_opening(self):
        result = render_template(
            "trading_decision/researchers/bull.jinja",
            speaking_as="Bull",
            opposing_speaker="Bear",
            debate_history="",
            opposing_last="",
            **_analysis_kwargs(),
        )
        assert "Bull Researcher" in result
        assert "AAPL" in result
        assert "long-term hold" in result
        # Opening-turn hint should appear when there's no opposing argument.
        assert "opening" in result.lower()

    def test_bull_rebuttal_includes_opposing(self):
        result = render_template(
            "trading_decision/researchers/bull.jinja",
            speaking_as="Bull",
            opposing_speaker="Bear",
            debate_history="Bull Researcher: opener\n\nBear Researcher: response",
            opposing_last="Bear Researcher: response",
            **_analysis_kwargs(),
        )
        assert "Bear Researcher: response" in result
        assert "rebut" in result.lower()

    def test_bear_opening(self):
        result = render_template(
            "trading_decision/researchers/bear.jinja",
            speaking_as="Bear",
            opposing_speaker="Bull",
            debate_history="",
            opposing_last="",
            **_analysis_kwargs(),
        )
        assert "Bear Researcher" in result
        assert "opening" in result.lower()

    def test_judge_synthesis(self):
        result = render_template(
            "trading_decision/researchers/investment_judge.jinja",
            debate_history="full transcript here",
            **_analysis_kwargs(),
        )
        assert "Investment Judge" in result
        assert "full transcript here" in result
        # Hold-reservation discipline.
        assert "do not default to hold" in result.lower()
        assert "InvestmentJudgeOutput" in result

    def test_analysis_context_partial_renders_each_field(self):
        result = render_template(
            "trading_decision/researchers/bull.jinja",
            speaking_as="Bull",
            opposing_speaker="Bear",
            debate_history="",
            opposing_last="",
            **_analysis_kwargs(
                market_regime={"regime": "expansion"},
                valuation={"signal": "cheap"},
                narrative="Some research notes.",
            ),
        )
        assert "regime" in result
        assert "expansion" in result
        assert "cheap" in result
        assert "Some research notes." in result


# ── Trader template ──────────────────────────────────────────────────────────


@pytest.mark.unit
class TestTraderPrompt:
    def test_renders_with_judge_input(self):
        result = render_template(
            "trading_decision/trader.jinja",
            investment_judge=_judge_payload(),
            **_analysis_kwargs(),
        )
        assert "Trader" in result
        assert "TraderOutput" in result
        # Judge payload is dumped as JSON in the prompt.
        assert "Net bull." in result
        # 3-tier action mapping table.
        assert "strong_buy" in result
        assert "hold" in result


# ── Risk debate templates ────────────────────────────────────────────────────


@pytest.mark.unit
class TestRiskDebatePrompts:
    def test_aggressive_opening(self):
        result = render_template(
            "trading_decision/risk_debate/aggressive.jinja",
            investment_judge=_judge_payload(),
            trader=_trader_payload(),
            risk_debate_history="",
            **_analysis_kwargs(),
        )
        assert "Aggressive Risk Analyst" in result
        # Inputs partial pulls in judge + trader.
        assert "Net bull." in result
        assert "Starter long." in result

    def test_conservative_with_history(self):
        result = render_template(
            "trading_decision/risk_debate/conservative.jinja",
            investment_judge=_judge_payload(),
            trader=_trader_payload(),
            risk_debate_history="Aggressive Analyst: press it",
            **_analysis_kwargs(),
        )
        assert "Conservative Risk Analyst" in result
        assert "Aggressive Analyst: press it" in result

    def test_neutral_opening(self):
        result = render_template(
            "trading_decision/risk_debate/neutral.jinja",
            investment_judge=_judge_payload(),
            trader=_trader_payload(),
            risk_debate_history="",
            **_analysis_kwargs(),
        )
        assert "Neutral Risk Analyst" in result


# ── Portfolio Manager template ───────────────────────────────────────────────


@pytest.mark.unit
class TestPortfolioManagerPrompt:
    def test_renders_full_inputs(self):
        result = render_template(
            "trading_decision/portfolio_manager.jinja",
            investment_judge=_judge_payload(),
            trader=_trader_payload(),
            risk_debate_history="Risk debate transcript here.",
            past_reflections="",
            **_analysis_kwargs(),
        )
        assert "Portfolio Manager" in result
        assert "PortfolioDecisionOutput" in result
        assert "Risk debate transcript here." in result
        # Hold-reservation discipline.
        assert "do not default to hold" in result.lower()
        # When past_reflections is empty, the section should not appear.
        assert "Past lessons" not in result

    def test_includes_past_reflections_when_provided(self):
        result = render_template(
            "trading_decision/portfolio_manager.jinja",
            investment_judge=_judge_payload(),
            trader=_trader_payload(),
            risk_debate_history="transcript",
            past_reflections="- **AAPL 2026-01-01** → buy | raw +5%",
            **_analysis_kwargs(),
        )
        assert "Past lessons" in result
        assert "AAPL 2026-01-01" in result


# ── Reflector template ───────────────────────────────────────────────────────


@pytest.mark.unit
class TestReflectorPrompt:
    def test_renders(self):
        result = render_template(
            "trading_decision/reflection/reflector.jinja",
            ticker="AAPL",
            decision_date="2026-05-17",
            decision={"rating": "buy", "executive_summary": "Buy AAPL."},
            outcome={"raw_return_pct": 5.0, "alpha_return_pct": 1.2, "holding_days": 5},
        )
        assert "Trading Reflector" in result
        assert "AAPL" in result
        assert "2026-05-17" in result
        # Decision + outcome dumped as JSON.
        assert "Buy AAPL." in result
        assert "alpha" in result.lower()
