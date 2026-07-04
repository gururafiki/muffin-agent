"""Unit tests for :class:`MemoryConfiguration` and :func:`resolve_user_id`."""

from __future__ import annotations

import pytest

from muffin_agent.utils.memory_config import (
    MEMORIES_NAMESPACE_ROOT,
    MemoryConfiguration,
    MemoryUnavailableError,
)


@pytest.mark.unit
class TestMemoriesNamespaceRoot:
    def test_value(self):
        assert MEMORIES_NAMESPACE_ROOT == ("memories",)


@pytest.mark.unit
class TestResolveUserId:
    """``MemoryConfiguration.resolve_user_id`` is the single source of truth
    for the configurable.user_id → memory_debug_user_id → fail fallback chain.
    """

    def test_returns_configurable_user_id_when_present(self):
        config = {"configurable": {"user_id": "alice"}}
        assert MemoryConfiguration.resolve_user_id(config) == "alice"

    def test_verified_identity_wins_over_client_user_id(self):
        # langgraph_auth_user_id is injected server-side from the verified
        # auth.py identity — a client-supplied user_id cannot spoof it.
        config = {
            "configurable": {
                "langgraph_auth_user_id": "verified-uuid",
                "user_id": "spoofed-victim",
            }
        }
        assert MemoryConfiguration.resolve_user_id(config) == "verified-uuid"

    @pytest.mark.parametrize("sentinel", ["anonymous", "api-client"])
    def test_sentinel_identities_fall_through(self, sentinel):
        config = {
            "configurable": {
                "langgraph_auth_user_id": sentinel,
                "user_id": "alice",
            }
        }
        assert MemoryConfiguration.resolve_user_id(config) == "alice"

    def test_sentinel_identity_reaches_debug_fallback(self, monkeypatch):
        monkeypatch.setenv("MEMORY_DEBUG_USER_ID", "shared")
        config = {"configurable": {"langgraph_auth_user_id": "anonymous"}}
        assert MemoryConfiguration.resolve_user_id(config) == "shared"

    def test_falls_back_to_memory_debug_user_id_env(self, monkeypatch):
        monkeypatch.setenv("MEMORY_DEBUG_USER_ID", "debug-alex")
        config = {"configurable": {}}
        assert MemoryConfiguration.resolve_user_id(config) == "debug-alex"

    def test_falls_back_to_memory_debug_user_id_configurable(self, monkeypatch):
        monkeypatch.delenv("MEMORY_DEBUG_USER_ID", raising=False)
        config = {"configurable": {"memory_debug_user_id": "debug-bob"}}
        assert MemoryConfiguration.resolve_user_id(config) == "debug-bob"

    def test_real_user_id_takes_precedence_over_debug_fallback(self, monkeypatch):
        monkeypatch.setenv("MEMORY_DEBUG_USER_ID", "debug-alex")
        config = {"configurable": {"user_id": "alice"}}
        assert MemoryConfiguration.resolve_user_id(config) == "alice"

    def test_returns_none_when_allow_missing_true(self, monkeypatch):
        monkeypatch.delenv("MEMORY_DEBUG_USER_ID", raising=False)
        config = {"configurable": {}}
        assert MemoryConfiguration.resolve_user_id(config, allow_missing=True) is None

    def test_raises_when_allow_missing_false_and_nothing_set(self, monkeypatch):
        monkeypatch.delenv("MEMORY_DEBUG_USER_ID", raising=False)
        config = {"configurable": {}}
        with pytest.raises(MemoryUnavailableError, match="user_id is not set"):
            MemoryConfiguration.resolve_user_id(config, allow_missing=False)

    def test_empty_string_user_id_falls_through(self, monkeypatch):
        """Empty-string ``user_id`` is treated as missing (not a valid identity)."""
        monkeypatch.delenv("MEMORY_DEBUG_USER_ID", raising=False)
        config = {"configurable": {"user_id": ""}}
        assert MemoryConfiguration.resolve_user_id(config, allow_missing=True) is None

    def test_non_string_user_id_falls_through(self, monkeypatch):
        """Non-string ``user_id`` (e.g. int) is treated as missing."""
        monkeypatch.delenv("MEMORY_DEBUG_USER_ID", raising=False)
        config = {"configurable": {"user_id": 42}}
        assert MemoryConfiguration.resolve_user_id(config, allow_missing=True) is None


@pytest.mark.unit
class TestMemoryUnavailableError:
    def test_is_lookup_error(self):
        """Subclass of LookupError so narrow handlers can catch it specifically."""
        assert issubclass(MemoryUnavailableError, LookupError)
