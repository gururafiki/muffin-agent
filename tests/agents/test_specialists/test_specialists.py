"""Tests for the specialist signal agents (technicals + sentiment) — v4."""

from __future__ import annotations

from typing import Any

import pytest

from muffin_agent.agents.specialists import (
    SentimentSignal,
    TechnicalSignal,
    build_sentiment_analysis_agent,
    build_technical_analysis_agent,
)
from muffin_agent.agents.specialists.sentiment_analysis import (
    compute_sentiment_signal_node,
)
from muffin_agent.agents.specialists.technical_analysis import (
    compute_technical_signal_node,
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
class TestSpecialistSubgraphs:
    """Both specialists compile into a deterministic 2-node StateGraph."""

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
