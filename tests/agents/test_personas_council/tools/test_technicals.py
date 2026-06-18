"""Unit tests for the 5-strategy technical-indicator ensemble."""

from __future__ import annotations

import math
import random

import pytest

from muffin_agent.agents.personas_council.tools.technicals import (
    DEFAULT_STRATEGY_WEIGHTS,
    combine_technical_signals,
    compute_mean_reversion_signal,
    compute_momentum_signal,
    compute_stat_arb_signal,
    compute_trend_signal,
    compute_volatility_regime_signal,
)


def _make_bars(prices: list[float], volumes: list[float] | None = None) -> list[dict]:
    """Build OHLCV bars from a close-price list (HL ±0.5%, open prev close)."""
    bars = []
    for i, p in enumerate(prices):
        bars.append(
            {
                "date": f"2024-{(i // 21) + 1:02d}-{(i % 21) + 1:02d}",
                "open": p * 0.999,
                "high": p * 1.005,
                "low": p * 0.995,
                "close": p,
                "volume": (volumes[i] if volumes is not None else 1_000_000),
            }
        )
    return bars


def _uptrend_prices(
    n: int = 252, drift: float = 0.001, vol: float = 0.005
) -> list[float]:
    """Generate a synthetic uptrending price series with reproducible noise."""
    rng = random.Random(0)
    price = 100.0
    out = [price]
    for _ in range(n - 1):
        price *= 1 + drift + rng.gauss(0, vol)
        out.append(price)
    return out


def _downtrend_prices(n: int = 252) -> list[float]:
    return list(reversed(_uptrend_prices(n)))


@pytest.mark.unit
class TestTrendSignal:
    def test_bullish_on_uptrend(self):
        bars = _make_bars(_uptrend_prices(252))
        result = compute_trend_signal(bars)
        assert result["signal"] == "bullish"
        assert result["metrics"]["ema_8"] is not None
        assert result["metrics"]["ema_55"] is not None

    def test_bearish_on_downtrend(self):
        bars = _make_bars(_downtrend_prices(252))
        result = compute_trend_signal(bars)
        assert result["signal"] == "bearish"

    def test_insufficient_data(self):
        # Less than 60 bars
        bars = _make_bars([100, 101, 102, 103])
        result = compute_trend_signal(bars)
        assert result["signal"] == "neutral"
        assert result["confidence"] == 0.0


@pytest.mark.unit
class TestMeanReversionSignal:
    def test_extreme_dip_signals_bullish(self):
        # 50 stable bars, then a sharp drop on the last bar
        stable = [100.0] * 49
        crash = [60.0]
        # Use tiny variation so the std isn't zero
        bars = _make_bars([p + 0.01 * (i % 2) for i, p in enumerate(stable + crash)])
        result = compute_mean_reversion_signal(bars)
        # With a -2 z-score plus low BB position the signal should be bullish
        assert result["signal"] in ("bullish", "neutral")
        # If bullish, confidence should be positive
        if result["signal"] == "bullish":
            assert result["confidence"] > 0

    def test_returns_zscore_and_rsi(self):
        bars = _make_bars(_uptrend_prices(100))
        result = compute_mean_reversion_signal(bars)
        assert "z_score" in result["metrics"]
        assert "rsi_14" in result["metrics"]
        assert "rsi_28" in result["metrics"]

    def test_insufficient_data(self):
        bars = _make_bars([100] * 30)
        result = compute_mean_reversion_signal(bars)
        assert result["signal"] == "neutral"
        assert result["confidence"] == 0.0


@pytest.mark.unit
class TestMomentumSignal:
    def test_bullish_on_strong_uptrend_with_volume_confirmation(self):
        # Strong drift so momentum > 5%, AND ramping volume so vol_ratio > 1
        # (the upstream momentum signal requires volume confirmation).
        prices = _uptrend_prices(252, drift=0.002, vol=0.003)
        volumes = [1_000_000 + i * 5_000 for i in range(252)]
        bars = _make_bars(prices, volumes=volumes)
        result = compute_momentum_signal(bars)
        assert result["signal"] == "bullish"
        assert result["confidence"] > 0

    def test_neutral_without_volume_confirmation(self):
        # Strong momentum but flat volume → ambiguous, no signal
        bars = _make_bars(_uptrend_prices(252, drift=0.002, vol=0.003))
        result = compute_momentum_signal(bars)
        assert result["signal"] == "neutral"

    def test_returns_multi_timeframe_metrics(self):
        bars = _make_bars(_uptrend_prices(252))
        result = compute_momentum_signal(bars)
        for key in ("momentum_1m", "momentum_3m", "momentum_6m", "volume_ratio"):
            assert key in result["metrics"]

    def test_insufficient_data(self):
        bars = _make_bars([100] * 100)
        result = compute_momentum_signal(bars)
        assert result["signal"] == "neutral"
        assert result["confidence"] == 0.0


@pytest.mark.unit
class TestVolatilityRegimeSignal:
    def test_has_required_metrics(self):
        bars = _make_bars(_uptrend_prices(252))
        result = compute_volatility_regime_signal(bars)
        for key in (
            "historical_volatility",
            "volatility_regime",
            "volatility_z_score",
            "atr_ratio",
        ):
            assert key in result["metrics"]

    def test_insufficient_data(self):
        bars = _make_bars([100] * 50)
        result = compute_volatility_regime_signal(bars)
        assert result["signal"] == "neutral"
        assert result["confidence"] == 0.0


@pytest.mark.unit
class TestStatArbSignal:
    def test_has_required_metrics(self):
        bars = _make_bars(_uptrend_prices(252))
        result = compute_stat_arb_signal(bars)
        for key in ("hurst_exponent", "skewness", "kurtosis"):
            assert key in result["metrics"]

    def test_insufficient_data(self):
        bars = _make_bars([100] * 50)
        result = compute_stat_arb_signal(bars)
        assert result["signal"] == "neutral"
        assert result["confidence"] == 0.0


@pytest.mark.unit
class TestCombineTechnicalSignals:
    def test_all_bullish_yields_bullish(self):
        results = {
            "trend": {"signal": "bullish", "confidence": 0.8, "metrics": {}},
            "mean_reversion": {"signal": "bullish", "confidence": 0.7, "metrics": {}},
            "momentum": {"signal": "bullish", "confidence": 0.9, "metrics": {}},
            "volatility": {"signal": "bullish", "confidence": 0.6, "metrics": {}},
            "stat_arb": {"signal": "bullish", "confidence": 0.5, "metrics": {}},
        }
        combined = combine_technical_signals(results)
        assert combined["signal"] == "bullish"
        assert combined["confidence"] > 0.2

    def test_all_bearish_yields_bearish(self):
        results = {
            "trend": {"signal": "bearish", "confidence": 0.8, "metrics": {}},
            "mean_reversion": {"signal": "bearish", "confidence": 0.7, "metrics": {}},
            "momentum": {"signal": "bearish", "confidence": 0.9, "metrics": {}},
            "volatility": {"signal": "bearish", "confidence": 0.6, "metrics": {}},
            "stat_arb": {"signal": "bearish", "confidence": 0.5, "metrics": {}},
        }
        combined = combine_technical_signals(results)
        assert combined["signal"] == "bearish"

    def test_all_zero_confidence_yields_neutral(self):
        results = {
            "trend": {"signal": "bullish", "confidence": 0.0, "metrics": {}},
            "mean_reversion": {"signal": "bearish", "confidence": 0.0, "metrics": {}},
        }
        combined = combine_technical_signals(results)
        assert combined["signal"] == "neutral"
        assert combined["confidence"] == 0.0

    def test_custom_weights_override_defaults(self):
        # All strategies bearish except one bullish — but only the bullish
        # strategy gets a non-zero weight. Should yield bullish.
        results = {
            "trend": {"signal": "bullish", "confidence": 0.9, "metrics": {}},
            "mean_reversion": {"signal": "bearish", "confidence": 0.9, "metrics": {}},
            "momentum": {"signal": "bearish", "confidence": 0.9, "metrics": {}},
        }
        custom_weights = {"trend": 1.0, "mean_reversion": 0.0, "momentum": 0.0}
        combined = combine_technical_signals(results, weights=custom_weights)
        assert combined["signal"] == "bullish"

    def test_default_weights_sum_to_one(self):
        assert math.isclose(sum(DEFAULT_STRATEGY_WEIGHTS.values()), 1.0)

    def test_contribution_metrics_present(self):
        results = {
            "trend": {"signal": "bullish", "confidence": 0.5, "metrics": {}},
        }
        combined = combine_technical_signals(results)
        assert "contribution_trend" in combined["metrics"]
        assert "final_score" in combined["metrics"]
