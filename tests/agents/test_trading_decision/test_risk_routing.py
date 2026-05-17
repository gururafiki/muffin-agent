"""Routing tests for the 3-way risk-debate alternation (PR 3)."""

import pytest

from muffin_agent.agents.trading_decision.conditional_logic import (
    DEFAULT_MAX_RISK_DEBATE_ROUNDS,
    _route_risk_debate,
)


def _state(*, count: int, latest_speaker: str = "") -> dict:
    return {
        "risk_debate": {
            "history": "",
            "aggressive_history": "",
            "conservative_history": "",
            "neutral_history": "",
            "current_aggressive_response": "",
            "current_conservative_response": "",
            "current_neutral_response": "",
            "latest_speaker": latest_speaker,
            "judge_decision": "",
            "count": count,
        }
    }


@pytest.mark.unit
class TestRouteRiskDebate:
    def test_empty_state_starts_with_aggressive(self):
        assert _route_risk_debate({}, {}) == "aggressive_debator"

    def test_opening_routes_to_aggressive(self):
        assert _route_risk_debate(_state(count=0), {}) == "aggressive_debator"

    def test_after_aggressive_routes_to_conservative(self):
        state = _state(count=1, latest_speaker="Aggressive")
        assert _route_risk_debate(state, {}) == "conservative_debator"

    def test_after_conservative_routes_to_neutral(self):
        state = _state(count=2, latest_speaker="Conservative")
        assert _route_risk_debate(state, {}) == "neutral_debator"

    def test_after_neutral_routes_back_to_aggressive(self):
        # Only reachable when max_risk_debate_rounds >= 2.
        state = _state(count=3, latest_speaker="Neutral")
        config = {"max_risk_debate_rounds": 2}
        assert _route_risk_debate(state, config) == "aggressive_debator"

    def test_default_exit_after_1_round(self):
        # Default = 1 round = 3 turns. Exit at count >= 3.
        state = _state(
            count=3 * DEFAULT_MAX_RISK_DEBATE_ROUNDS,
            latest_speaker="Neutral",
        )
        assert _route_risk_debate(state, {}) == "portfolio_manager"

    def test_one_below_default_exit_still_routes(self):
        state = _state(
            count=3 * DEFAULT_MAX_RISK_DEBATE_ROUNDS - 1,
            latest_speaker="Conservative",
        )
        assert _route_risk_debate(state, {}) == "neutral_debator"

    @pytest.mark.parametrize("rounds,expected_exit_count", [(1, 3), (2, 6), (3, 9)])
    def test_exit_threshold_scales_with_rounds(
        self, rounds: int, expected_exit_count: int
    ):
        configurable = {"max_risk_debate_rounds": rounds}
        # one turn short of exit → still routing
        state = _state(count=expected_exit_count - 1, latest_speaker="Conservative")
        assert _route_risk_debate(state, configurable) == "neutral_debator"
        # at exit threshold → portfolio manager
        state = _state(count=expected_exit_count, latest_speaker="Neutral")
        assert _route_risk_debate(state, configurable) == "portfolio_manager"

    def test_full_default_round_trace(self):
        """Walk through a default 1-round debate verifying every transition."""
        # Turn 0: open with Aggressive.
        assert _route_risk_debate(_state(count=0), {}) == "aggressive_debator"
        # Turn 1: after Aggressive, Conservative.
        assert (
            _route_risk_debate(_state(count=1, latest_speaker="Aggressive"), {})
            == "conservative_debator"
        )
        # Turn 2: after Conservative, Neutral.
        assert (
            _route_risk_debate(_state(count=2, latest_speaker="Conservative"), {})
            == "neutral_debator"
        )
        # Turn 3: exit to Portfolio Manager.
        assert (
            _route_risk_debate(_state(count=3, latest_speaker="Neutral"), {})
            == "portfolio_manager"
        )

    def test_portfolio_manager_latest_routes_to_aggressive(self):
        """If the PM is somehow re-entered (testing edge case), restart cycle."""
        state = _state(count=0, latest_speaker="Portfolio Manager")
        assert _route_risk_debate(state, {}) == "aggressive_debator"
