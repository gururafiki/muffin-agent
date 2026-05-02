"""Tests for the ``CollectionFindings`` schema."""

import pytest
from pydantic import ValidationError

from muffin_agent.middlewares.subagent_refinement import (
    CollectionFindings,
    Gap,
    GapReason,
)


@pytest.mark.unit
class TestCollectionFindings:
    def test_minimal_valid(self):
        f = CollectionFindings(call_id="abc")
        assert f.call_id == "abc"
        assert f.confidence == 1.0
        assert f.gaps == []

    def test_full_round_trip(self):
        payload = {
            "call_id": "xyz",
            "requested": ["pe", "ev_ebitda"],
            "obtained": {"pe": 12.3},
            "gaps": [
                {
                    "field": "ev_ebitda",
                    "reason": "no_data",
                    "detail": "AMZN missing",
                    "retry_advice": "give_up",
                }
            ],
            "confidence": 0.5,
            "tools_used": [
                {"tool": "equity_price", "status": "success"},
                {
                    "tool": "equity_estimates_forward_eps",
                    "status": "error",
                    "error_class": "HTTP 422",
                },
            ],
            "notes": "FMP free tier",
        }
        f = CollectionFindings.model_validate(payload)
        assert len(f.gaps) == 1
        assert f.gaps[0].reason == GapReason.NO_DATA
        round_tripped = CollectionFindings.model_validate_json(f.model_dump_json())
        assert round_tripped == f

    def test_gap_reason_rejects_unknown_value(self):
        with pytest.raises(ValidationError):
            Gap(field="x", reason="not_a_real_reason")

    def test_confidence_bounds(self):
        with pytest.raises(ValidationError):
            CollectionFindings(call_id="abc", confidence=1.5)
        with pytest.raises(ValidationError):
            CollectionFindings(call_id="abc", confidence=-0.1)
