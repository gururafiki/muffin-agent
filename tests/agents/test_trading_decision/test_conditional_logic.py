"""Routing tests for the investment debate alternation logic.

Routing is split into two functions:

* ``_route_investment_debate(state, configurable)`` — pure logic, used by
  these unit tests.
* ``should_continue_investment_debate(state)`` — the graph-facing
  wrapper that pulls the active ``RunnableConfig`` via
  ``langgraph.config.get_config()``. End-to-end coverage of the wrapper
  lives in ``test_graph.py``.
"""

import pytest

from muffin_agent.agents.trading_decision.conditional_logic import (
    BEAR_TAG,
    BULL_TAG,
    DEFAULT_MAX_INVESTMENT_DEBATE_ROUNDS,
    _route_investment_debate,
)


def _state(*, count: int, current: str = "") -> dict:
    return {
        "investment_debate": {
            "history": "",
            "bull_history": "",
            "bear_history": "",
            "current_response": current,
            "judge_decision": "",
            "count": count,
        }
    }


@pytest.mark.unit
class TestRouteInvestmentDebate:
    def test_empty_state_starts_with_bull(self):
        assert _route_investment_debate({}, {}) == "bull_researcher"

    def test_opening_debate_goes_to_bull(self):
        # No prior turn, count = 0 → Bull opens.
        assert _route_investment_debate(_state(count=0), {}) == "bull_researcher"

    def test_after_bull_routes_to_bear(self):
        state = _state(count=1, current=f"{BULL_TAG} our case is X.")
        assert _route_investment_debate(state, {}) == "bear_researcher"

    def test_after_bear_routes_to_bull(self):
        state = _state(count=1, current=f"{BEAR_TAG} our case is Y.")
        assert _route_investment_debate(state, {}) == "bull_researcher"

    def test_default_exit_after_2_rounds(self):
        # Default = 2 rounds = 4 turns. count must be >= 4 to exit.
        state = _state(
            count=2 * DEFAULT_MAX_INVESTMENT_DEBATE_ROUNDS,
            current=f"{BEAR_TAG} ...",
        )
        assert _route_investment_debate(state, {}) == "investment_judge"

    def test_one_below_default_exit_still_routes(self):
        state = _state(
            count=2 * DEFAULT_MAX_INVESTMENT_DEBATE_ROUNDS - 1,
            current=f"{BULL_TAG} ...",
        )
        assert _route_investment_debate(state, {}) == "bear_researcher"

    @pytest.mark.parametrize("rounds,expected_exit_count", [(1, 2), (2, 4), (3, 6)])
    def test_exit_threshold_scales_with_rounds(
        self, rounds: int, expected_exit_count: int
    ):
        configurable = {"max_investment_debate_rounds": rounds}
        # one turn short of exit → still routing
        state = _state(count=expected_exit_count - 1, current=f"{BULL_TAG} ...")
        assert _route_investment_debate(state, configurable) == "bear_researcher"
        # at exit threshold → judge
        state = _state(count=expected_exit_count, current=f"{BULL_TAG} ...")
        assert _route_investment_debate(state, configurable) == "investment_judge"

    def test_alternation_for_full_2_round_debate(self):
        """Walk through a default 2-round debate verifying every transition."""
        # Round 1: Bull → Bear
        state = _state(count=0)
        assert _route_investment_debate(state, {}) == "bull_researcher"

        state = _state(count=1, current=f"{BULL_TAG} round 1 bull")
        assert _route_investment_debate(state, {}) == "bear_researcher"

        # Round 2 (rebuttal): Bull → Bear
        state = _state(count=2, current=f"{BEAR_TAG} round 1 bear")
        assert _route_investment_debate(state, {}) == "bull_researcher"

        state = _state(count=3, current=f"{BULL_TAG} round 2 bull rebuttal")
        assert _route_investment_debate(state, {}) == "bear_researcher"

        # Final: exit to judge
        state = _state(count=4, current=f"{BEAR_TAG} round 2 bear rebuttal")
        assert _route_investment_debate(state, {}) == "investment_judge"
