"""Tests for the ToolErrorHandler middleware."""

import pytest

from muffin_agent.agents.data_collection.utils import (
    PERMANENT_ERROR_PATTERNS,
    _cache_key,
    _is_permanent_error,
)


@pytest.mark.unit
class TestIsPermanentError:
    """Test permanent error classification."""

    @pytest.mark.parametrize(
        "msg",
        [
            "Missing credential 'intrinio_api_key'",
            "HTTP error 404: Not Found",
            "unauthorized access",
            "Error 403: Forbidden",
            "api_key is required",
            "Provider not supported for this endpoint",
            "Invalid parameter 'foo'",
            "Authentication failed",
        ],
    )
    def test_permanent_errors_detected(self, msg):
        assert _is_permanent_error(msg) is True

    @pytest.mark.parametrize(
        "msg",
        [
            "Connection timed out",
            "Rate limit exceeded",
            "Internal server error",
            "HTTP error 500",
            "Network unreachable",
        ],
    )
    def test_transient_errors_not_flagged(self, msg):
        assert _is_permanent_error(msg) is False

    def test_case_insensitive(self):
        assert _is_permanent_error("MISSING CREDENTIAL 'foo'") is True
        assert _is_permanent_error("UNAUTHORIZED") is True

    def test_patterns_list_not_empty(self):
        assert len(PERMANENT_ERROR_PATTERNS) > 0


@pytest.mark.unit
class TestCacheKey:
    """Test cache key generation."""

    def test_same_args_same_key(self):
        call1 = {"name": "tool_a", "args": {"x": 1, "y": 2}, "id": "id1"}
        call2 = {"name": "tool_a", "args": {"x": 1, "y": 2}, "id": "id2"}
        assert _cache_key(call1) == _cache_key(call2)

    def test_different_arg_order_same_key(self):
        call1 = {"name": "tool_a", "args": {"y": 2, "x": 1}, "id": "id1"}
        call2 = {"name": "tool_a", "args": {"x": 1, "y": 2}, "id": "id2"}
        assert _cache_key(call1) == _cache_key(call2)

    def test_different_args_different_key(self):
        call1 = {"name": "tool_a", "args": {"x": 1}, "id": "id1"}
        call2 = {"name": "tool_a", "args": {"x": 2}, "id": "id2"}
        assert _cache_key(call1) != _cache_key(call2)

    def test_different_tool_different_key(self):
        call1 = {"name": "tool_a", "args": {"x": 1}, "id": "id1"}
        call2 = {"name": "tool_b", "args": {"x": 1}, "id": "id2"}
        assert _cache_key(call1) != _cache_key(call2)

    def test_missing_args_uses_empty_dict(self):
        call = {"name": "tool_a", "id": "id1"}
        key = _cache_key(call)
        assert key == "tool_a:{}"

    def test_id_ignored(self):
        call1 = {"name": "tool_a", "args": {}, "id": "abc"}
        call2 = {"name": "tool_a", "args": {}, "id": "xyz"}
        assert _cache_key(call1) == _cache_key(call2)
