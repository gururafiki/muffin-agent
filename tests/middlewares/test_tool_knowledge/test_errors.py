"""Tests for the error classifier + duplicate-block key."""

import pytest

from muffin_agent.middlewares.tool_knowledge.errors import (
    duplicate_key,
    is_permanent_error,
)


@pytest.mark.unit
class TestIsPermanentError:
    @pytest.mark.parametrize(
        "msg",
        [
            "Missing credential 'intrinio_api_key'",
            "HTTP 422: limit must be less than or equal to 5",
            "HTTP 404: Not Found",
            "Unauthorized request",
            "Invalid parameter 'foo'",
            "No estimates data was returned for AMZN",
        ],
    )
    def test_permanent_hints(self, msg):
        assert is_permanent_error(msg) is True

    @pytest.mark.parametrize(
        "msg",
        [
            "HTTP 502 Bad Gateway",
            "Connection reset",
            "Timed out after 30s",
            "Internal server error",
        ],
    )
    def test_transient_strings_are_not_permanent(self, msg):
        assert is_permanent_error(msg) is False

    def test_non_string(self):
        assert is_permanent_error(None) is False  # type: ignore[arg-type]


@pytest.mark.unit
class TestDuplicateKey:
    def test_same_args_same_key(self):
        a = duplicate_key({"name": "tool_a", "args": {"x": 1, "y": 2}})
        b = duplicate_key({"name": "tool_a", "args": {"y": 2, "x": 1}})
        assert a == b

    def test_different_args_different_key(self):
        a = duplicate_key({"name": "tool_a", "args": {"x": 1}})
        b = duplicate_key({"name": "tool_a", "args": {"x": 2}})
        assert a != b

    def test_different_tools_different_key(self):
        a = duplicate_key({"name": "tool_a", "args": {}})
        b = duplicate_key({"name": "tool_b", "args": {}})
        assert a != b
