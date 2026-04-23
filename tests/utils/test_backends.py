"""Unit tests for ``muffin_agent.utils.backends``."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


@pytest.mark.unit
class TestMakeAgentBackend:
    def test_default_routes_wired(self, tmp_path):
        """``/skills/``, ``/scratch/``, ``/memories/`` route to expected backends."""
        from muffin_agent.utils.backends import make_agent_backend

        runtime = MagicMock()
        runtime.config = {"configurable": {"thread_id": "t", "user_id": "alice"}}

        with patch("muffin_agent.utils.backends.get_backend") as mock_get_backend:
            mock_get_backend.return_value = MagicMock(name="sandbox")
            factory = make_agent_backend(skills_root=tmp_path)
            composite = factory(runtime)

        assert "/skills/" in composite.routes
        assert "/scratch/" in composite.routes
        assert "/memories/" in composite.routes
        assert composite.default is mock_get_backend.return_value

    def test_custom_skills_root(self, tmp_path):
        """``skills_root`` parameter plumbs through to ``FilesystemBackend``."""
        from muffin_agent.utils.backends import make_agent_backend

        runtime = MagicMock()
        runtime.config = {"configurable": {"thread_id": "t"}}
        custom_root = tmp_path / "custom_skills"
        custom_root.mkdir()

        with patch("muffin_agent.utils.backends.get_backend") as mock_get_backend:
            mock_get_backend.return_value = MagicMock(name="sandbox")
            factory = make_agent_backend(skills_root=custom_root)
            composite = factory(runtime)

        skills_fs = composite.routes["/skills/"]
        # FilesystemBackend stores its root under ``cwd`` (resolved absolute path).
        assert Path(skills_fs.cwd) == custom_root.resolve()

    def test_default_instance_is_callable(self):
        """``get_agent_backend`` is a ready-to-use ``BackendFactory``."""
        from muffin_agent.utils.backends import get_agent_backend

        assert callable(get_agent_backend)


@pytest.mark.unit
class TestMemoriesNamespace:
    """Tests for ``_memories_namespace``.

    Config is read via :func:`langgraph.config.get_config` (patched in
    each test), not from the ``rt`` argument — the runtime parameter is
    intentionally unused, matching the ``NamespaceFactory`` contract.
    The ``rt`` arg here is just ``MagicMock()``.
    """

    def test_reads_configurable_user_id(self):
        """``user_id`` is read from ``get_config()['configurable']``."""
        from muffin_agent.utils.backends import _memories_namespace

        with patch(
            "muffin_agent.utils.backends.get_config",
            return_value={"configurable": {"user_id": "alice"}},
        ):
            assert _memories_namespace(MagicMock()) == ("memories", "alice")

    def test_raises_when_user_id_missing(self, monkeypatch):
        """Missing ``user_id`` raises :class:`MemoryUnavailableError`."""
        from muffin_agent.utils.backends import (
            MemoryUnavailableError,
            _memories_namespace,
        )

        monkeypatch.delenv("MEMORY_DEBUG_USER_ID", raising=False)
        with patch(
            "muffin_agent.utils.backends.get_config",
            return_value={"configurable": {}},
        ):
            with pytest.raises(MemoryUnavailableError, match="user_id is not set"):
                _memories_namespace(MagicMock())

    def test_raises_when_user_id_empty_or_non_string(self, monkeypatch):
        """Empty-string or non-string ``user_id`` raises ``MemoryUnavailableError``."""
        from muffin_agent.utils.backends import (
            MemoryUnavailableError,
            _memories_namespace,
        )

        monkeypatch.delenv("MEMORY_DEBUG_USER_ID", raising=False)
        for bad in ("", 42, None):
            with patch(
                "muffin_agent.utils.backends.get_config",
                return_value={"configurable": {"user_id": bad}},
            ):
                with pytest.raises(MemoryUnavailableError, match="user_id is not set"):
                    _memories_namespace(MagicMock())

    def test_memory_unavailable_error_is_lookup_error(self):
        """``MemoryUnavailableError`` is catchable as ``LookupError``.

        The middleware distinguishes it from other exceptions, but
        broader handlers (tool-error handler) treat it uniformly.
        """
        from muffin_agent.utils.backends import MemoryUnavailableError

        assert issubclass(MemoryUnavailableError, LookupError)

    def test_falls_back_to_memory_debug_user_id_env(self, monkeypatch):
        """``MEMORY_DEBUG_USER_ID`` env var serves as a debug fallback."""
        from muffin_agent.utils.backends import _memories_namespace

        monkeypatch.setenv("MEMORY_DEBUG_USER_ID", "debug-alex")
        with patch(
            "muffin_agent.utils.backends.get_config",
            return_value={"configurable": {}},
        ):
            assert _memories_namespace(MagicMock()) == ("memories", "debug-alex")

    def test_env_fallback_applies_when_get_config_raises(self, monkeypatch):
        """Env fallback works when ``get_config()`` raises (no graph context).

        Regression: deepagents calls the namespace callback during
        background runs / store operations where no graph execution
        context is active.  ``get_config()`` raises ``RuntimeError``
        there — the env fallback must still fire.
        """
        from muffin_agent.utils.backends import _memories_namespace

        monkeypatch.setenv("MEMORY_DEBUG_USER_ID", "debug-alex")
        with patch(
            "muffin_agent.utils.backends.get_config",
            side_effect=RuntimeError("no graph context"),
        ):
            assert _memories_namespace(MagicMock()) == ("memories", "debug-alex")

    def test_falls_back_to_memory_debug_user_id_configurable(self, monkeypatch):
        """``configurable.memory_debug_user_id`` also serves as a debug fallback."""
        from muffin_agent.utils.backends import _memories_namespace

        monkeypatch.delenv("MEMORY_DEBUG_USER_ID", raising=False)
        with patch(
            "muffin_agent.utils.backends.get_config",
            return_value={"configurable": {"memory_debug_user_id": "debug-bob"}},
        ):
            assert _memories_namespace(MagicMock()) == ("memories", "debug-bob")

    def test_real_user_id_takes_precedence_over_debug_fallback(self, monkeypatch):
        """Real ``configurable.user_id`` wins over the debug fallback."""
        from muffin_agent.utils.backends import _memories_namespace

        monkeypatch.setenv("MEMORY_DEBUG_USER_ID", "debug-alex")
        with patch(
            "muffin_agent.utils.backends.get_config",
            return_value={"configurable": {"user_id": "alice"}},
        ):
            assert _memories_namespace(MagicMock()) == ("memories", "alice")
