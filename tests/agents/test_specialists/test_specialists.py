"""Tests for the specialist signal agents (technicals + sentiment)."""

from __future__ import annotations

from typing import Any

import pytest

from muffin_agent.agents.specialists import (
    SPECIALIST_REGISTRY,
    SentimentSignal,
    TechnicalSignal,
    build_single_specialist_graph,
    sentiment_analysis_node,
    technical_analysis_node,
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


def _downtrend_bars(n: int = 252) -> list[dict[str, Any]]:
    return list(reversed(_uptrend_bars(n)))


@pytest.mark.unit
class TestSpecialistRegistry:
    def test_contains_both(self):
        assert set(SPECIALIST_REGISTRY) == {"technicals", "sentiment"}

    def test_each_has_signal_schema(self):
        for spec in SPECIALIST_REGISTRY.values():
            assert spec.signal_schema is not None
            assert hasattr(spec.signal_schema, "model_validate")


@pytest.mark.unit
@pytest.mark.asyncio
class TestTechnicalAnalysisNode:
    async def test_uptrend_yields_bullish(self):
        bundle = {"prices_1y": _uptrend_bars(252)}
        result = await technical_analysis_node({"data_bundle": bundle}, {})
        sig = result["persona_signals"][0]
        assert sig["agent_id"] == "technicals"
        assert sig["signal"] in ("buy", "strong_buy")
        # Evidence carries per-strategy + weighted results
        ev = sig["evidence"]
        expected_keys = (
            "trend",
            "mean_reversion",
            "momentum",
            "volatility_regime",
            "stat_arb",
            "weighted",
        )
        for key in expected_keys:
            assert key in ev

    async def test_missing_bundle_returns_hold(self):
        result = await technical_analysis_node({"ticker": "ZAB"}, {})
        sig = result["persona_signals"][0]
        assert sig["signal"] == "hold"
        assert sig["confidence"] == 0.0

    async def test_error_bundle_returns_hold(self):
        result = await technical_analysis_node(
            {"data_bundle": {"error": "timeout"}}, {}
        )
        sig = result["persona_signals"][0]
        assert sig["signal"] == "hold"

    async def test_short_price_series_returns_hold(self):
        bundle = {"prices_1y": _uptrend_bars(10)}
        result = await technical_analysis_node({"data_bundle": bundle}, {})
        sig = result["persona_signals"][0]
        assert sig["signal"] == "hold"

    async def test_signal_validates_against_schema(self):
        bundle = {"prices_1y": _uptrend_bars(252)}
        result = await technical_analysis_node({"data_bundle": bundle}, {})
        sig = result["persona_signals"][0]
        # Round-trip into TechnicalSignal
        validated = TechnicalSignal.model_validate(sig)
        assert validated.agent_id == "technicals"


@pytest.mark.unit
@pytest.mark.asyncio
class TestSentimentAnalysisNode:
    async def test_bullish_alignment(self):
        bundle = {
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
        result = await sentiment_analysis_node({"data_bundle": bundle}, {})
        sig = result["persona_signals"][0]
        assert sig["agent_id"] == "sentiment"
        assert sig["signal"] in ("buy", "strong_buy")

    async def test_news_dominates_due_to_weighting(self):
        # 3 insider buys (weight 0.3 → 0.9) vs 5 news negatives (weight 0.7 → 3.5)
        bundle = {
            "insider_trades": [{"transaction_shares": 100} for _ in range(3)],
            "company_news": [{"sentiment": "negative"} for _ in range(5)],
        }
        result = await sentiment_analysis_node({"data_bundle": bundle}, {})
        sig = result["persona_signals"][0]
        assert sig["signal"] in ("sell", "strong_sell")

    async def test_missing_bundle_returns_hold(self):
        result = await sentiment_analysis_node({"ticker": "ZAB"}, {})
        sig = result["persona_signals"][0]
        assert sig["signal"] == "hold"
        assert sig["confidence"] == 0.0

    async def test_empty_inputs_returns_hold(self):
        bundle = {"insider_trades": [], "company_news": []}
        result = await sentiment_analysis_node({"data_bundle": bundle}, {})
        sig = result["persona_signals"][0]
        assert sig["signal"] == "hold"

    async def test_signal_validates_against_schema(self):
        bundle = {
            "insider_trades": [{"transaction_shares": 100}],
            "company_news": [{"sentiment": "positive"}],
        }
        result = await sentiment_analysis_node({"data_bundle": bundle}, {})
        sig = result["persona_signals"][0]
        validated = SentimentSignal.model_validate(sig)
        assert validated.agent_id == "sentiment"


@pytest.mark.unit
class TestSingleSpecialistGraph:
    def test_compiles_for_each_specialist(self):
        for slug in SPECIALIST_REGISTRY:
            g = build_single_specialist_graph(slug)
            nodes = list(g.get_graph().nodes)
            assert "persona_data_collection" in nodes
            assert slug in nodes

    def test_unknown_slug_raises(self):
        with pytest.raises(KeyError, match="Unknown specialist slug"):
            build_single_specialist_graph("not_a_real_specialist")
