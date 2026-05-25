"""Schema tests for the PR 4 reflection-memory types."""

import pytest
from pydantic import ValidationError

from muffin_agent.agents.trading_decision.schemas import (
    DecisionRecord,
    Outcome,
)


@pytest.mark.unit
class TestOutcome:
    def test_minimal_valid(self):
        out = Outcome(raw_return_pct=5.3, alpha_return_pct=1.2, holding_days=5)
        assert out.benchmark == "SPY"
        assert out.decision_action is None

    def test_holding_days_must_be_positive(self):
        with pytest.raises(ValidationError):
            Outcome(raw_return_pct=1.0, alpha_return_pct=0.5, holding_days=0)

    def test_custom_benchmark_and_action(self):
        out = Outcome(
            raw_return_pct=-2.1,
            alpha_return_pct=-0.4,
            holding_days=10,
            benchmark="QQQ",
            decision_action="hold",
        )
        assert out.benchmark == "QQQ"
        assert out.decision_action == "hold"

    def test_round_trip_json(self):
        out = Outcome(raw_return_pct=5.3, alpha_return_pct=1.2, holding_days=5)
        rehydrated = Outcome.model_validate(out.model_dump())
        assert rehydrated == out


@pytest.mark.unit
class TestDecisionRecord:
    def test_pending_minimal(self):
        rec = DecisionRecord(
            ticker="AAPL",
            date="2026-05-17",
            status="pending",
            decision={"rating": "buy"},
        )
        assert rec.outcome is None
        assert rec.reflection is None

    def test_resolved_full(self):
        rec = DecisionRecord(
            ticker="AAPL",
            date="2026-05-17",
            status="resolved",
            decision={"rating": "buy"},
            outcome=Outcome(raw_return_pct=5.3, alpha_return_pct=1.2, holding_days=5),
            reflection="Bull thesis on services growth held; alpha +1.2%.",
        )
        assert rec.status == "resolved"
        assert rec.outcome is not None
        assert rec.outcome.holding_days == 5
        assert rec.reflection is not None and "alpha" in rec.reflection

    def test_status_must_be_in_enum(self):
        with pytest.raises(ValidationError):
            DecisionRecord(
                ticker="AAPL",
                date="2026-05-17",
                status="archived",  # invalid
                decision={"rating": "buy"},
            )

    def test_round_trip_resolved(self):
        rec = DecisionRecord(
            ticker="AAPL",
            date="2026-05-17",
            status="resolved",
            decision={"rating": "buy"},
            outcome=Outcome(raw_return_pct=5.3, alpha_return_pct=1.2, holding_days=5),
            reflection="text",
        )
        rehydrated = DecisionRecord.model_validate(rec.model_dump())
        assert rehydrated == rec
