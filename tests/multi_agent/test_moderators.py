"""Tests for the moderator implementations."""

from __future__ import annotations

import pytest

from muffin_agent.multi_agent import (
    AlternatingModerator,
    RoundRobinModerator,
    Turn,
)


@pytest.mark.unit
class TestRoundRobinModerator:
    def test_opens_with_first_speaker(self):
        mod = RoundRobinModerator(["alpha", "beta", "gamma"])
        assert mod.next_speaker({}) == "alpha"

    def test_cycles_through_in_order(self):
        mod = RoundRobinModerator(["alpha", "beta", "gamma"])
        turns: list[Turn] = []
        names: list[str] = []
        for _ in range(7):
            name = mod.next_speaker({"transcript": list(turns)})
            names.append(name)
            turns.append({"speaker": name, "content": "", "round": 1})
        assert names == ["alpha", "beta", "gamma", "alpha", "beta", "gamma", "alpha"]

    def test_index_modulo_handles_partial_rounds(self):
        mod = RoundRobinModerator(["a", "b", "c"])
        # 4 prior turns → index = 4 % 3 = 1 → "b"
        state = {"transcript": [{"speaker": "x", "content": "", "round": 1}] * 4}
        assert mod.next_speaker(state) == "b"


@pytest.mark.unit
class TestAlternatingModerator:
    def test_opens_with_speaker_a(self):
        mod = AlternatingModerator("bull", "bear")
        assert mod.next_speaker({}) == "bull"

    def test_after_a_routes_to_b(self):
        mod = AlternatingModerator("bull", "bear")
        state = {"transcript": [{"speaker": "bull", "content": "", "round": 1}]}
        assert mod.next_speaker(state) == "bear"

    def test_after_b_routes_back_to_a(self):
        mod = AlternatingModerator("bull", "bear")
        state = {
            "transcript": [
                {"speaker": "bull", "content": "", "round": 1},
                {"speaker": "bear", "content": "", "round": 1},
            ]
        }
        assert mod.next_speaker(state) == "bull"

    def test_lead_count_breaks_tie_to_a_when_equal(self):
        mod = AlternatingModerator("bull", "bear")
        # Equal counts → a wins the tie.
        state = {
            "transcript": [
                {"speaker": "bull", "content": "", "round": 1},
                {"speaker": "bear", "content": "", "round": 1},
                {"speaker": "bull", "content": "", "round": 2},
                {"speaker": "bear", "content": "", "round": 2},
            ]
        }
        assert mod.next_speaker(state) == "bull"

    def test_ignores_non_participant_turns(self):
        # If state contains turns from unknown speakers, they should not
        # affect the alternation count.
        mod = AlternatingModerator("bull", "bear")
        state = {
            "transcript": [
                {"speaker": "moderator", "content": "intro", "round": 1},
                {"speaker": "bull", "content": "open", "round": 1},
            ]
        }
        # 1 bull, 0 bear → bear is next.
        assert mod.next_speaker(state) == "bear"
