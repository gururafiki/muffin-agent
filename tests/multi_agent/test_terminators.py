"""Tests for the terminator implementations."""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage

from muffin_agent.multi_agent import MaxRoundsTerminator


def _msgs(n: int) -> list:
    return [AIMessage(content="", name="x") for _ in range(n)]


@pytest.mark.unit
class TestMaxRoundsTerminator:
    def test_empty_state_continues(self):
        term = MaxRoundsTerminator(max_rounds=2, num_participants=3)
        stop, reason = term.should_stop({})
        assert stop is False
        assert reason is None

    def test_below_threshold_continues(self):
        term = MaxRoundsTerminator(max_rounds=2, num_participants=3)
        # 5 messages < 2 * 3 = 6
        stop, _ = term.should_stop({"messages": _msgs(5)})
        assert stop is False

    def test_at_threshold_stops(self):
        term = MaxRoundsTerminator(max_rounds=2, num_participants=3)
        # 6 messages == 2 * 3 → stop
        stop, reason = term.should_stop({"messages": _msgs(6)})
        assert stop is True
        assert reason is not None and "max_rounds=2" in reason

    def test_above_threshold_stops(self):
        term = MaxRoundsTerminator(max_rounds=1, num_participants=2)
        stop, _ = term.should_stop({"messages": _msgs(10)})
        assert stop is True

    def test_one_round_one_participant(self):
        term = MaxRoundsTerminator(max_rounds=1, num_participants=1)
        assert term.should_stop({})[0] is False
        assert term.should_stop({"messages": _msgs(1)})[0] is True
