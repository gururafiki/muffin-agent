"""Tests for the specialist signal agents (technicals + sentiment) — v4."""

from __future__ import annotations

from typing import Any

import pytest

from muffin_agent.agents.personas_council.specialists import (
    FundamentalsSignal,
    GrowthSignal,
    NewsSentimentSignal,
    SentimentSignal,
    TechnicalSignal,
    ValuationSignal,
    build_sentiment_analysis_agent,
    build_technical_analysis_agent,
)
from muffin_agent.agents.personas_council.specialists.fundamentals_analysis import (
    compute_fundamentals_signal_node,
)
from muffin_agent.agents.personas_council.specialists.growth_analysis import (
    compute_growth_signal_node,
)
from muffin_agent.agents.personas_council.specialists.news_sentiment_analysis import (
    aggregate_sentiment_node,
)
from muffin_agent.agents.personas_council.specialists.sentiment_analysis import (
    compute_sentiment_signal_node,
)
from muffin_agent.agents.personas_council.specialists.technical_analysis import (
    compute_technical_signal_node,
)
from muffin_agent.agents.personas_council.specialists.valuation_analysis import (
    compute_valuation_signal_node,
)


def _uptrend_bars(n: int = 252) -> list[dict[str, Any]]:
    """Generate a synthetic uptrend OHLCV series with ramping volume."""
    bars = []
    for i in range(n):
        p = 100 * (1.003**i)
        bars.append(
            {
                "date": f"2025-{(i // 21) + 1:02d}-{(i % 21) + 1:02d}",
                "open": p * 0.999,
                "high": p * 1.005,
                "low": p * 0.995,
                "close": p,
                "volume": 1_000_000 + i * 5_000,
            }
        )
    return bars


@pytest.mark.unit
class TestTechnicalCompute:
    """Pure-Python compute_technical_signal_node tests (no MCP / no graph)."""

    def test_uptrend_yields_bullish(self):
        state = {"prices_1y": _uptrend_bars(252)}
        result = compute_technical_signal_node(state)
        sig = result["persona_signals"][0]
        assert sig["agent_id"] == "technicals"
        assert sig["signal"] in ("buy", "strong_buy")
        # Evidence carries per-strategy + weighted results
        ev = sig["evidence"]
        for key in (
            "trend",
            "mean_reversion",
            "momentum",
            "volatility_regime",
            "stat_arb",
            "weighted",
        ):
            assert key in ev

    def test_empty_prices_returns_hold(self):
        result = compute_technical_signal_node({"prices_1y": []})
        sig = result["persona_signals"][0]
        assert sig["signal"] == "hold"
        assert sig["confidence"] == 0.0

    def test_short_price_series_returns_hold(self):
        result = compute_technical_signal_node({"prices_1y": _uptrend_bars(10)})
        sig = result["persona_signals"][0]
        assert sig["signal"] == "hold"

    def test_signal_validates_against_schema(self):
        result = compute_technical_signal_node({"prices_1y": _uptrend_bars(252)})
        sig = result["persona_signals"][0]
        validated = TechnicalSignal.model_validate(sig)
        assert validated.agent_id == "technicals"


@pytest.mark.unit
class TestSentimentCompute:
    """Pure-Python compute_sentiment_signal_node tests (no MCP / no graph)."""

    def test_bullish_alignment(self):
        state = {
            "insider_trades": [
                {"transaction_shares": 1_000},
                {"transaction_shares": 500},
                {"transaction_shares": -200},
            ],
            "company_news": [
                {"sentiment": "positive"},
                {"sentiment": "positive"},
                {"sentiment": "positive"},
                {"sentiment": "negative"},
            ],
        }
        result = compute_sentiment_signal_node(state)
        sig = result["persona_signals"][0]
        assert sig["agent_id"] == "sentiment"
        assert sig["signal"] in ("buy", "strong_buy")

    def test_news_dominates_due_to_weighting(self):
        # 3 insider buys (weight 0.3) vs 5 news negatives (weight 0.7)
        state = {
            "insider_trades": [{"transaction_shares": 100} for _ in range(3)],
            "company_news": [{"sentiment": "negative"} for _ in range(5)],
        }
        result = compute_sentiment_signal_node(state)
        sig = result["persona_signals"][0]
        assert sig["signal"] in ("sell", "strong_sell")

    def test_empty_inputs_returns_hold(self):
        state = {"insider_trades": [], "company_news": []}
        result = compute_sentiment_signal_node(state)
        sig = result["persona_signals"][0]
        assert sig["signal"] == "hold"
        assert sig["confidence"] == 0.0

    def test_signal_validates_against_schema(self):
        state = {
            "insider_trades": [{"transaction_shares": 100}],
            "company_news": [{"sentiment": "positive"}],
        }
        result = compute_sentiment_signal_node(state)
        sig = result["persona_signals"][0]
        validated = SentimentSignal.model_validate(sig)
        assert validated.agent_id == "sentiment"


@pytest.mark.unit
class TestFundamentalsCompute:
    """compute_fundamentals_signal_node (ported fundamentals specialist)."""

    def test_strong_cheap_company_is_bullish(self):
        state = {
            "ticker": "AAPL",
            "return_on_equity": 0.25,
            "net_margin": 0.25,
            "operating_margin": 0.25,
            "revenue_growth": 0.20,
            "earnings_growth": 0.20,
            "book_value_growth": 0.15,
            "current_ratio": 2.0,
            "debt_to_equity": 0.3,
            "free_cash_flow_per_share": 5.0,
            "earnings_per_share": 4.0,
            "price_to_earnings_ratio": 12,
            "price_to_book_ratio": 2,
            "price_to_sales_ratio": 1,
        }
        sig = compute_fundamentals_signal_node(state)["persona_signals"][0]
        assert sig["agent_id"] == "fundamentals"
        assert sig["signal"] in ("buy", "strong_buy")
        FundamentalsSignal.model_validate(sig)

    def test_no_metrics_returns_hold(self):
        sig = compute_fundamentals_signal_node({"ticker": "X"})["persona_signals"][0]
        assert sig["signal"] == "hold"
        assert sig["confidence"] == 0.0


@pytest.mark.unit
class TestGrowthCompute:
    """compute_growth_signal_node (ported growth specialist)."""

    def test_accelerating_growth_is_bullish(self):
        state = {
            "ticker": "NVDA",
            "revenue_growth_history": [0.10, 0.18, 0.25],
            "eps_growth_history": [0.12, 0.20, 0.28],
            "fcf_growth_history": [0.05, 0.12, 0.20],
            "gross_margin_history": [0.45, 0.50, 0.55],
            "operating_margin_history": [0.12, 0.15, 0.18],
            "net_margin_history": [0.08, 0.10, 0.12],
            "peg_ratio": 0.8,
            "price_to_sales_ratio": 1.5,
            "debt_to_equity": 0.3,
            "current_ratio": 2.5,
            "insider_trades": [
                {"transaction_shares": 1000, "transaction_value": 50_000}
            ],
        }
        sig = compute_growth_signal_node(state)["persona_signals"][0]
        assert sig["agent_id"] == "growth"
        assert sig["signal"] in ("buy", "strong_buy")
        GrowthSignal.model_validate(sig)

    def test_empty_state_handled(self):
        sig = compute_growth_signal_node({"ticker": "X"})["persona_signals"][0]
        assert sig["agent_id"] == "growth"
        GrowthSignal.model_validate(sig)


@pytest.mark.unit
class TestValuationCompute:
    """compute_valuation_signal_node (ported valuation specialist)."""

    def test_undervalued_is_bullish(self):
        state = {
            "ticker": "BRK",
            "market_cap": 100.0,
            "net_income_latest": 50.0,
            "depreciation_latest": 10.0,
            "capital_expenditure_latest": 5.0,
            "working_capital_history": [10.0, 10.0],
            "earnings_growth": 0.10,
            "revenue_growth": 0.12,
            "free_cash_flow_history": [40.0, 45.0, 50.0],
            "total_debt_latest": 20.0,
            "cash_latest": 30.0,
            "interest_coverage_latest": 10.0,
            "debt_to_equity_latest": 0.3,
            "enterprise_value_latest": 120.0,
            "ev_to_ebitda_history": [8.0, 9.0, 7.0],
            "price_to_book_ratio_latest": 2.0,
            "book_value_growth": 0.05,
        }
        sig = compute_valuation_signal_node(state)["persona_signals"][0]
        assert sig["agent_id"] == "valuation"
        assert sig["signal"] in ("buy", "strong_buy")
        ValuationSignal.model_validate(sig)

    def test_capex_abs_and_wc_delta(self):
        # negative capex should be abs()'d; ΔWC from the last two periods
        state = {
            "ticker": "X",
            "market_cap": 1000.0,
            "net_income_latest": 100.0,
            "depreciation_latest": 20.0,
            "capital_expenditure_latest": -30.0,
            "working_capital_history": [50.0, 60.0],
            "free_cash_flow_history": [80.0, 90.0],
            "earnings_growth": 0.1,
            "revenue_growth": 0.1,
            "total_debt_latest": 0.0,
            "cash_latest": 0.0,
            "interest_coverage_latest": None,
            "debt_to_equity_latest": None,
            "enterprise_value_latest": None,
            "ev_to_ebitda_history": None,
            "price_to_book_ratio_latest": None,
            "book_value_growth": None,
        }
        sig = compute_valuation_signal_node(state)["persona_signals"][0]
        ValuationSignal.model_validate(sig)


@pytest.mark.unit
class TestNewsSentimentAggregate:
    """aggregate_sentiment_node (ported LLM news_sentiment specialist)."""

    def test_majority_positive_is_bullish(self):
        state = {
            "ticker": "AAPL",
            "articles": [
                {"title": "a", "sentiment": "positive", "confidence": 0.9},
                {"title": "b", "sentiment": "positive", "confidence": 0.8},
                {"title": "c", "sentiment": "negative", "confidence": 0.6},
            ],
        }
        sig = aggregate_sentiment_node(state)["persona_signals"][0]
        assert sig["agent_id"] == "news_sentiment"
        assert sig["signal"] in ("buy", "strong_buy")
        NewsSentimentSignal.model_validate(sig)

    def test_no_articles_returns_hold(self):
        sig = aggregate_sentiment_node({"ticker": "X", "articles": []})[
            "persona_signals"
        ][0]
        assert sig["signal"] == "hold"
        assert sig["confidence"] == 0.0

    def test_confidence_blends_llm_and_proportion(self):
        # 2 positive (conf 1.0) vs 0 negative → proportion 1.0, conf = 0.7*1+0.3*1=1.0
        state = {
            "ticker": "X",
            "articles": [
                {"title": "a", "sentiment": "positive", "confidence": 1.0},
                {"title": "b", "sentiment": "positive", "confidence": 1.0},
            ],
        }
        sig = aggregate_sentiment_node(state)["persona_signals"][0]
        assert sig["confidence"] == pytest.approx(1.0)


@pytest.mark.unit
class TestSpecialistSubgraphs:
    """Deterministic specialists compile into 2-/3-node StateGraphs."""

    def test_technical_analysis_compiles(self):
        g = build_technical_analysis_agent()
        nodes = list(g.get_graph().nodes)
        assert "fetch_ohlcv" in nodes
        assert "compute_technical_signal" in nodes

    def test_sentiment_analysis_compiles(self):
        g = build_sentiment_analysis_agent()
        nodes = list(g.get_graph().nodes)
        assert "fetch_insider_trades" in nodes
        assert "fetch_company_news" in nodes
        assert "compute_sentiment_signal" in nodes
