"""Regression tests for stanley_druckenmiller numeric coercion.

``StanleyDruckenmillerRawData.prices_1y`` is ``list[dict[str, Any]]``, and
Pydantic does not coerce those inner values, so a weak LLM can extract ``close``
as a string. Both price-reading scorers here used to build their series with a
bare ``[b.get("close") for b in prices if b.get("close") is not None]``, which

  * raised ``TypeError: '>' not supported between instances of 'str' and 'int'``
    on ``closes[i - 1] > 0`` in ``_score_druckenmiller_risk_reward``, and
  * fed strings into ``compute_price_momentum(prices: list[float])``.

This is the identical failure that took down the AMZN council run through
``nassim_taleb`` (thread 019f8476-…). It was still latent here, and mypy is what
surfaced it — the scorers now go through the shared ``clean_series`` guard.
"""

from __future__ import annotations

import pytest

from muffin_agent.agents.personas_council.personas.stanley_druckenmiller import (
    _score_druckenmiller_growth,
    _score_druckenmiller_risk_reward,
)


def _bars(closes: list[object]) -> list[dict[str, object]]:
    return [{"close": c} for c in closes]


@pytest.mark.unit
class TestStringClosesDoNotCrash:
    def test_risk_reward_survives_string_closes(self):
        # 25 bars so the >= 20 volatility branch actually runs.
        state = {"prices_1y": _bars([str(100 + i) for i in range(25)])}
        result = _score_druckenmiller_risk_reward(state)  # type: ignore[arg-type]
        assert result.daily_volatility is not None

    def test_growth_survives_string_closes(self):
        state = {"prices_1y": _bars([str(100 + i) for i in range(25)])}
        _evidence, momentum_pct = _score_druckenmiller_growth(state)  # type: ignore[arg-type]
        # 100 -> 124 over the window is a real, positive return.
        assert momentum_pct is not None
        assert momentum_pct > 0

    def test_non_numeric_closes_are_dropped_not_fatal(self):
        state = {"prices_1y": _bars([None, "n/a", {}, True, "100", 110, 99.5])}
        result = _score_druckenmiller_risk_reward(state)  # type: ignore[arg-type]
        # Only the 3 genuinely numeric bars survive, which is below the 20-bar
        # threshold, so volatility stays unset rather than blowing up.
        assert result.daily_volatility is None

    def test_mixed_string_and_numeric_closes(self):
        state = {"prices_1y": _bars(["100", 101, "102.5", 103] * 7)}
        result = _score_druckenmiller_risk_reward(state)  # type: ignore[arg-type]
        assert result.daily_volatility is not None
