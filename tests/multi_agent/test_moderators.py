"""Tests for the moderator implementations."""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage

from muffin_agent.multi_agent import (
    AlternatingModerator,
    RoundRobinModerator,
)


def _msg(name: str) -> AIMessage:
    return AIMessage(content="", name=name)


@pytest.mark.unit
class TestRoundRobinModerator:
    def test_opens_with_first_speaker(self):
        mod = RoundRobinModerator(["alpha", "beta", "gamma"])
        assert mod.next_speaker({}) == "alpha"

    def test_cycles_through_in_order(self):
        mod = RoundRobinModerator(["alpha", "beta", "gamma"])
        msgs: list = []
        names: list[str] = []
        for _ in range(7):
            name = mod.next_speaker({"messages": list(msgs)})
            names.append(name)
            msgs.append(_msg(name))
        assert names == ["alpha", "beta", "gamma", "alpha", "beta", "gamma", "alpha"]

    def test_index_modulo_handles_partial_rounds(self):
        mod = RoundRobinModerator(["a", "b", "c"])
        # 4 prior messages → index = 4 % 3 = 1 → "b"
        state = {"messages": [_msg("x")] * 4}
        assert mod.next_speaker(state) == "b"


@pytest.mark.unit
class TestAlternatingModerator:
    def test_opens_with_speaker_a(self):
        mod = AlternatingModerator("bull", "bear")
        assert mod.next_speaker({}) == "bull"

    def test_after_a_routes_to_b(self):
        mod = AlternatingModerator("bull", "bear")
        state = {"messages": [_msg("bull")]}
        assert mod.next_speaker(state) == "bear"

    def test_after_b_routes_back_to_a(self):
        mod = AlternatingModerator("bull", "bear")
        state = {"messages": [_msg("bull"), _msg("bear")]}
        assert mod.next_speaker(state) == "bull"

    def test_lead_count_breaks_tie_to_a_when_equal(self):
        mod = AlternatingModerator("bull", "bear")
        state = {
            "messages": [
                _msg("bull"),
                _msg("bear"),
                _msg("bull"),
                _msg("bear"),
            ]
        }
        assert mod.next_speaker(state) == "bull"

    def test_ignores_non_participant_messages(self):
        mod = AlternatingModerator("bull", "bear")
        state = {
            "messages": [
                _msg("moderator"),
                _msg("bull"),
            ]
        }
        # 1 bull, 0 bear → bear is next.
        assert mod.next_speaker(state) == "bear"
