"""Unit tests for persona schemas (v4)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from muffin_agent.agents.personas import AnalystSignal


@pytest.mark.unit
class TestAnalystSignal:
    def test_minimal_construction(self):
        s = AnalystSignal(
            agent_id="warren_buffett",
            signal="strong_buy",
            confidence=0.85,
            reasoning="Wide moat, generous owner earnings, 20% MOS.",
        )
        assert s.evidence == {}

    def test_signal_must_be_5_tier(self):
        with pytest.raises(ValidationError):
            AnalystSignal(
                agent_id="x",
                signal="bullish",  # type: ignore[arg-type]  # not a 5-tier value
                confidence=0.5,
                reasoning="r",
            )

    def test_confidence_must_be_bounded(self):
        with pytest.raises(ValidationError):
            AnalystSignal(agent_id="x", signal="buy", confidence=1.5, reasoning="r")
        with pytest.raises(ValidationError):
            AnalystSignal(agent_id="x", signal="buy", confidence=-0.1, reasoning="r")

    def test_evidence_is_freeform_by_default(self):
        s = AnalystSignal(
            agent_id="x",
            signal="hold",
            confidence=0.5,
            reasoning="r",
            evidence={"custom_field": 42, "nested": {"k": "v"}},
        )
        assert s.evidence["custom_field"] == 42
