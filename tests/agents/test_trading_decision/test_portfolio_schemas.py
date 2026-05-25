"""Schema validation tests for ``PortfolioDecisionOutput`` (PR 3)."""

import pytest
from pydantic import ValidationError

from muffin_agent.agents.trading_decision.schemas import PortfolioDecisionOutput


def _payload(**overrides) -> dict:
    base = {
        "rating": "buy",
        "executive_summary": (
            "Buy AAPL at 2% NAV starter; primary risk is China demand."
        ),
        "investment_thesis": (
            "Bull case held after Conservative's strongest objection. Trader's "
            "stop at 178.50 anchored to ex-ante VaR; sizing trimmed from 3% to "
            "2% to acknowledge the Conservative case on regulatory drag."
        ),
        "price_target": 220.0,
        "stop_loss": 178.5,
        "time_horizon": "3–6 months",
        "position_sizing": "2% of NAV starter, scale to 4% on Q1 beat",
        "key_risks_remaining": ["China demand", "Ad-tech regulation"],
        "confidence": 0.65,
    }
    base.update(overrides)
    return base


@pytest.mark.unit
class TestPortfolioDecisionOutput:
    def test_minimal_valid(self):
        out = PortfolioDecisionOutput(**_payload())
        assert out.rating == "buy"
        assert out.price_target == 220.0
        assert out.confidence == 0.65
        assert out.incorporates_past_lessons is False  # PR 4 flag stays False

    def test_rating_must_be_5_tier_enum(self):
        for valid in ("strong_sell", "sell", "hold", "buy", "strong_buy"):
            PortfolioDecisionOutput(**_payload(rating=valid))
        for invalid in ("overweight", "underweight", "watch"):
            with pytest.raises(ValidationError):
                PortfolioDecisionOutput(**_payload(rating=invalid))

    def test_optional_price_fields_default_to_none(self):
        payload = _payload()
        del payload["price_target"]
        del payload["stop_loss"]
        out = PortfolioDecisionOutput(**payload)
        assert out.price_target is None
        assert out.stop_loss is None

    def test_required_fields(self):
        for required in (
            "rating",
            "executive_summary",
            "investment_thesis",
            "time_horizon",
            "position_sizing",
            "confidence",
        ):
            payload = _payload()
            del payload[required]
            with pytest.raises(ValidationError):
                PortfolioDecisionOutput(**payload)

    def test_confidence_bounds(self):
        with pytest.raises(ValidationError):
            PortfolioDecisionOutput(**_payload(confidence=-0.1))
        with pytest.raises(ValidationError):
            PortfolioDecisionOutput(**_payload(confidence=1.1))

    def test_key_risks_remaining_defaults_to_empty(self):
        payload = _payload()
        del payload["key_risks_remaining"]
        out = PortfolioDecisionOutput(**payload)
        assert out.key_risks_remaining == []

    def test_round_trip_json(self):
        out = PortfolioDecisionOutput(**_payload())
        rehydrated = PortfolioDecisionOutput.model_validate(out.model_dump())
        assert rehydrated == out

    def test_incorporates_past_lessons_can_be_set(self):
        out = PortfolioDecisionOutput(**_payload(incorporates_past_lessons=True))
        assert out.incorporates_past_lessons is True
