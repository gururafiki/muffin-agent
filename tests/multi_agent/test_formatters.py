"""Tests for the pure transcript formatters."""

from __future__ import annotations

import pytest

from muffin_agent.multi_agent import (
    Turn,
    last_opposing_turn,
    render_transcript_chronological,
)


@pytest.mark.unit
class TestRenderTranscriptChronological:
    def test_empty(self):
        assert render_transcript_chronological([]) == ""

    def test_single_turn(self):
        turn: Turn = {"speaker": "alice", "content": "hello", "round": 1}
        assert render_transcript_chronological([turn]) == "alice: hello"

    def test_multiple_turns_chronological_order(self):
        turns: list[Turn] = [
            {"speaker": "alice", "content": "open", "round": 1},
            {"speaker": "bob", "content": "rebut", "round": 1},
            {"speaker": "alice", "content": "press", "round": 2},
        ]
        assert render_transcript_chronological(turns) == (
            "alice: open\n\nbob: rebut\n\nalice: press"
        )


@pytest.mark.unit
class TestLastOpposingTurn:
    def test_empty(self):
        assert last_opposing_turn([], "alice") is None

    def test_no_opposing_speakers(self):
        turns: list[Turn] = [
            {"speaker": "alice", "content": "monologue", "round": 1}
        ]
        assert last_opposing_turn(turns, "alice") is None

    def test_returns_most_recent_other_speaker(self):
        turns: list[Turn] = [
            {"speaker": "bob", "content": "first", "round": 1},
            {"speaker": "alice", "content": "rebut", "round": 1},
            {"speaker": "carol", "content": "last", "round": 1},
        ]
        result = last_opposing_turn(turns, "alice")
        assert result is not None
        assert result["speaker"] == "carol"
        assert result["content"] == "last"

    def test_skips_self_turns(self):
        turns: list[Turn] = [
            {"speaker": "bob", "content": "bob-says", "round": 1},
            {"speaker": "alice", "content": "alice-says", "round": 2},
        ]
        result = last_opposing_turn(turns, "alice")
        assert result is not None
        assert result["speaker"] == "bob"
