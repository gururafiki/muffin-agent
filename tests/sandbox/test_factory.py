"""Unit tests for SandboxFactory and public factory functions."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

_PATCH_PREFIX = "muffin_agent.sandbox.factory"


def _make_mock_config():
    """Build a mock Configuration."""
    config = MagicMock()
    config.opensandbox_url = "localhost:8080"
    config.opensandbox_api_key = None
    config.opensandbox_image = "python:3.11-slim"
    return config


def _make_paged_result(sandbox_infos=None):
    """Build a mock PagedSandboxInfos."""
    result = MagicMock()
    result.sandbox_infos = sandbox_infos or []
    return result


def _make_sandbox_info(sandbox_id="sb-123"):
    """Build a mock SandboxInfo."""
    info = MagicMock()
    info.id = sandbox_id
    return info


# ---------------------------------------------------------------------------
# Tests — SandboxFactory._get_thread_id
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSandboxFactoryThreadId:
    def test_from_runtime_config(self):
        from muffin_agent.sandbox.factory import SandboxFactory

        runtime = MagicMock()
        runtime.config = {"configurable": {"thread_id": "thread-abc"}}

        factory = SandboxFactory(runtime)
        assert factory._get_thread_id() == "thread-abc"

    def test_falls_back_to_get_config(self):
        from muffin_agent.sandbox.factory import SandboxFactory

        # Runtime without config attr (like Runtime from langgraph.runtime)
        runtime = MagicMock(spec=[])

        with patch(
            f"{_PATCH_PREFIX}.get_config",
            return_value={"configurable": {"thread_id": "thread-xyz"}},
        ):
            factory = SandboxFactory(runtime)
            assert factory._get_thread_id() == "thread-xyz"

    def test_defaults_to_default(self):
        from muffin_agent.sandbox.factory import SandboxFactory

        runtime = MagicMock()
        runtime.config = {"configurable": {}}

        factory = SandboxFactory(runtime)
        assert factory._get_thread_id() == "default"


# ---------------------------------------------------------------------------
# Tests — SandboxFactory.get_sandbox (sync)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSandboxFactoryGetSandbox:
    def test_connects_when_found(self):
        from muffin_agent.sandbox.factory import SandboxFactory

        runtime = MagicMock()
        runtime.config = {"configurable": {"thread_id": "thread-abc"}}

        mock_sandbox = MagicMock()
        mock_sandbox.id = "found-sb"

        mock_manager = MagicMock()
        mock_manager.list_sandbox_infos.return_value = _make_paged_result(
            [_make_sandbox_info("found-sb")],
        )

        with (
            patch(
                f"{_PATCH_PREFIX}.Configuration.from_runnable_config",
                return_value=_make_mock_config(),
            ),
            patch(
                "opensandbox.sync.manager.SandboxManagerSync.create",
                return_value=mock_manager,
            ),
            patch(
                "opensandbox.sync.sandbox.SandboxSync.connect",
                return_value=mock_sandbox,
            ) as mock_connect,
        ):
            factory = SandboxFactory(runtime)
            result = factory.get_sandbox()

        mock_connect.assert_called_once()
        assert mock_connect.call_args.args[0] == "found-sb"
        assert result.id == "found-sb"

    def test_creates_when_not_found(self):
        from muffin_agent.sandbox.factory import SandboxFactory

        runtime = MagicMock()
        runtime.config = {"configurable": {"thread_id": "thread-abc"}}

        mock_sandbox = MagicMock()
        mock_sandbox.id = "new-sb"

        mock_manager = MagicMock()
        mock_manager.list_sandbox_infos.return_value = _make_paged_result([])

        with (
            patch(
                f"{_PATCH_PREFIX}.Configuration.from_runnable_config",
                return_value=_make_mock_config(),
            ),
            patch(
                "opensandbox.sync.manager.SandboxManagerSync.create",
                return_value=mock_manager,
            ),
            patch(
                "opensandbox.sync.sandbox.SandboxSync.create",
                return_value=mock_sandbox,
            ) as mock_create,
        ):
            factory = SandboxFactory(runtime)
            result = factory.get_sandbox()

        mock_create.assert_called_once()
        assert result.id == "new-sb"
        # Verify thread_id metadata is passed
        assert mock_create.call_args.kwargs["metadata"] == {"thread_id": "thread-abc"}


# ---------------------------------------------------------------------------
# Tests — SandboxFactory.aget_sandbox (async)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSandboxFactoryAgetSandbox:
    @pytest.mark.asyncio
    async def test_connects_when_found(self):
        from muffin_agent.sandbox.factory import SandboxFactory

        runtime = MagicMock()
        runtime.config = {"configurable": {"thread_id": "thread-abc"}}

        mock_sandbox = MagicMock()
        mock_sandbox.id = "found-sb"

        mock_manager = MagicMock()
        mock_manager.list_sandbox_infos = AsyncMock(
            return_value=_make_paged_result([_make_sandbox_info("found-sb")]),
        )
        mock_manager.close = AsyncMock()

        with (
            patch(
                f"{_PATCH_PREFIX}.Configuration.from_runnable_config",
                return_value=_make_mock_config(),
            ),
            patch(
                "opensandbox.manager.SandboxManager.create",
                AsyncMock(return_value=mock_manager),
            ),
            patch(
                "opensandbox.sandbox.Sandbox.connect",
                AsyncMock(return_value=mock_sandbox),
            ) as mock_connect,
        ):
            factory = SandboxFactory(runtime)
            result = await factory.aget_sandbox()

        mock_connect.assert_called_once()
        assert mock_connect.call_args.args[0] == "found-sb"
        assert result.id == "found-sb"

    @pytest.mark.asyncio
    async def test_creates_when_not_found(self):
        from muffin_agent.sandbox.factory import SandboxFactory

        runtime = MagicMock()
        runtime.config = {"configurable": {"thread_id": "thread-abc"}}

        mock_sandbox = MagicMock()
        mock_sandbox.id = "new-sb"

        mock_manager = MagicMock()
        mock_manager.list_sandbox_infos = AsyncMock(
            return_value=_make_paged_result([]),
        )
        mock_manager.close = AsyncMock()

        with (
            patch(
                f"{_PATCH_PREFIX}.Configuration.from_runnable_config",
                return_value=_make_mock_config(),
            ),
            patch(
                "opensandbox.manager.SandboxManager.create",
                AsyncMock(return_value=mock_manager),
            ),
            patch(
                "opensandbox.sandbox.Sandbox.create",
                AsyncMock(return_value=mock_sandbox),
            ) as mock_create,
        ):
            factory = SandboxFactory(runtime)
            result = await factory.aget_sandbox()

        mock_create.assert_called_once()
        assert result.id == "new-sb"
        assert mock_create.call_args.kwargs["metadata"] == {"thread_id": "thread-abc"}


# ---------------------------------------------------------------------------
# Tests — get_backend() public function
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestGetBackend:
    def test_returns_backend_wrapping_sandbox(self):
        from muffin_agent.sandbox.factory import get_backend

        runtime = MagicMock()
        runtime.config = {"configurable": {"thread_id": "thread-abc"}}

        mock_sandbox = MagicMock()
        mock_sandbox.id = "sb-456"

        with patch(
            f"{_PATCH_PREFIX}.SandboxFactory.get_sandbox",
            return_value=mock_sandbox,
        ):
            backend = get_backend(runtime)

        assert backend.id == "sb-456"

    def test_works_with_runtime_no_config(self):
        """get_backend works when runtime has no config attribute (Runtime)."""
        from muffin_agent.sandbox.factory import get_backend

        mock_sandbox = MagicMock()
        mock_sandbox.id = "sb-789"

        # Simulate Runtime object — no 'config' attribute
        runtime = MagicMock(spec=[])

        with patch(
            f"{_PATCH_PREFIX}.SandboxFactory.get_sandbox",
            return_value=mock_sandbox,
        ):
            backend = get_backend(runtime)

        assert backend.id == "sb-789"
