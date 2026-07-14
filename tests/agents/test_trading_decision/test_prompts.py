"""Prompt-template rendering tests for trading_decision Jinja files.

Verifies that:
* Each per-role template renders without errors with the standard set of
  per-call vars.
* The shared partials (``_analyst_reports.jinja``,
  ``_risk_synthesis_inputs.jinja``) are pulled in correctly via
  ``{% include %}``.
* Hold-reservation discipline appears in the synthesis prompts.
"""

from __future__ import annotations

from typing import Any

import pytest

from muffin_agent.prompts import render_template

# ── Common kwargs ─────────────────────────────────────────────────────────────


def _analysis_kwargs(**overrides) -> dict[str, Any]:
    """Flat-field kwargs the downstream Jinja templates now consume."""
    base = {
        "ticker": "AAPL",
        "query": "long-term hold",
        "narrative": None,
        "market_report": None,
        "fundamentals_report": None,
        "news_report": None,
        "sentiment_report": None,
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
        # Conference participant convention: the shared transcript is
        # rendered into the system prompt as `transcript`.
        result = render_template(
            "trading_decision/investment_debate/bull.jinja",
            transcript="",
            **_analysis_kwargs(),
        )
        assert "Bull Researcher" in result
        assert "AAPL" in result
        assert "long-term hold" in result
        # Opening-turn hint should appear when the transcript is empty.
        assert "opening" in result.lower()

    def test_bull_rebuttal_includes_transcript(self):
        result = render_template(
            "trading_decision/investment_debate/bull.jinja",
            transcript="bull_researcher: opener\n\nbear_researcher: response",
            **_analysis_kwargs(),
        )
        assert "bear_researcher: response" in result
        assert "rebut" in result.lower()

    def test_bear_opening(self):
        result = render_template(
            "trading_decision/investment_debate/bear.jinja",
            transcript="",
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

    def test_analyst_reports_partial_renders_each_field(self):
        result = render_template(
            "trading_decision/investment_debate/bull.jinja",
            transcript="",
            **_analysis_kwargs(
                market_report="ATR 2.5 / RSI 62",
                fundamentals_report="ROIC 28%, FCF margin 22%",
                news_report="Q1 earnings 2026-04-25",
                sentiment_report="r/wallstreetbets cautiously bullish",
                narrative="Some research notes.",
            ),
        )
        assert "ATR 2.5 / RSI 62" in result
        assert "ROIC 28%" in result
        assert "Q1 earnings 2026-04-25" in result
        assert "r/wallstreetbets" in result
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
            transcript="",
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
            transcript="aggressive_debator: press it",
            **_analysis_kwargs(),
        )
        assert "Conservative Risk Analyst" in result
        assert "aggressive_debator: press it" in result

    def test_neutral_opening(self):
        result = render_template(
            "trading_decision/risk_debate/neutral.jinja",
            investment_judge=_judge_payload(),
            trader=_trader_payload(),
            transcript="",
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
            transcript="Risk debate transcript here.",
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
            transcript="transcript",
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


# ── Analyst templates ────────────────────────────────────────────────────────


@pytest.mark.unit
class TestAnalystPrompts:
    def test_market_analyst_lists_indicators_and_ticker(self):
        result = render_template(
            "trading_decision/analysts/market.jinja",
            ticker="AAPL",
            decision_date="2026-05-23",
        )
        assert "Market Analyst" in result
        assert "AAPL" in result
        assert "2026-05-23" in result
        # All supported indicator keys appear in the menu.
        for ind in (
            "close_50_sma",
            "close_200_sma",
            "macd",
            "macds",
            "macdh",
            "rsi",
            "boll",
            "boll_ub",
            "boll_lb",
            "atr",
            "vwma",
        ):
            assert ind in result

    def test_fundamentals_analyst_lists_tools(self):
        result = render_template(
            "trading_decision/analysts/fundamentals.jinja",
            ticker="AAPL",
            decision_date="2026-05-23",
        )
        assert "Fundamentals Analyst" in result
        for mcp in (
            "equity_fundamental_income",
            "equity_fundamental_balance",
            "equity_fundamental_cash",
            "equity_fundamental_ratios",
            "equity_fundamental_metrics",
        ):
            assert mcp in result

    def test_news_analyst_lists_news_tools(self):
        result = render_template(
            "trading_decision/analysts/news.jinja",
            ticker="AAPL",
            decision_date="2026-05-23",
        )
        assert "News & Macro Analyst" in result
        assert "news_company" in result
        assert "news_world" in result
        assert "equity_ownership_insider_trading" in result

    def test_social_analyst_mentions_sources(self):
        result = render_template(
            "trading_decision/analysts/social.jinja",
            ticker="AAPL",
            decision_date="2026-05-23",
        )
        assert "Social Sentiment Analyst" in result
        assert "news_company" in result
        assert "firecrawl_search" in result
        # Source-breakdown guidance.
        assert "Reddit" in result


# ── Shared `_analyst_reports.jinja` partial ──────────────────────────────────


@pytest.mark.unit
class TestAnalystReportsPartial:
    def test_renders_all_four_when_provided(self):
        result = render_template(
            "trading_decision/_analyst_reports.jinja",
            market_report="M",
            fundamentals_report="F",
            news_report="N",
            sentiment_report="S",
        )
        assert "Market analysis report" in result
        assert "Fundamentals report" in result
        assert "News & macro report" in result
        assert "Social sentiment report" in result
        # Bodies appear.
        for body in ("M", "F", "N", "S"):
            assert body in result

    def test_omits_missing_reports(self):
        result = render_template(
            "trading_decision/_analyst_reports.jinja",
            market_report="M",
        )
        assert "Market analysis report" in result
        assert "Fundamentals report" not in result
        assert "News & macro report" not in result
        assert "Social sentiment report" not in result
