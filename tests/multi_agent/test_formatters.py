"""Tests for the pure message-list formatters."""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from muffin_agent.multi_agent import (
    last_opposing_message,
    render_messages_chronological,
)


@pytest.mark.unit
class TestRenderMessagesChronological:
    def test_empty(self):
        assert render_messages_chronological([]) == ""

    def test_single_message_with_name(self):
        msgs = [AIMessage(content="hello", name="alice")]
        assert render_messages_chronological(msgs) == "alice: hello"

    def test_falls_back_to_class_name_when_no_name(self):
        msgs = [HumanMessage(content="hi there")]
        assert render_messages_chronological(msgs) == "HumanMessage: hi there"

    def test_multiple_messages_chronological_order(self):
        msgs = [
            AIMessage(content="open", name="alice"),
            AIMessage(content="rebut", name="bob"),
            AIMessage(content="press", name="alice"),
        ]
        assert render_messages_chronological(msgs) == (
            "alice: open\n\nbob: rebut\n\nalice: press"
        )


@pytest.mark.unit
class TestLastOpposingMessage:
    def test_empty(self):
        assert last_opposing_message([], "alice") is None

    def test_no_opposing_speakers(self):
        msgs = [AIMessage(content="monologue", name="alice")]
        assert last_opposing_message(msgs, "alice") is None

    def test_returns_most_recent_other_speaker(self):
        msgs = [
            AIMessage(content="first", name="bob"),
            AIMessage(content="rebut", name="alice"),
            AIMessage(content="last", name="carol"),
        ]
        result = last_opposing_message(msgs, "alice")
        assert result is not None
        assert result.name == "carol"
        assert result.content == "last"

    def test_skips_self_messages(self):
        msgs = [
            AIMessage(content="bob-says", name="bob"),
            AIMessage(content="alice-says", name="alice"),
        ]
        result = last_opposing_message(msgs, "alice")
        assert result is not None
        assert result.name == "bob"
