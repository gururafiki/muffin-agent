"""Sandbox factory for muffin agents.

Encapsulates sandbox discovery by ``thread_id`` metadata and lazy creation.
Works with both ``ToolRuntime`` (tools) and ``Runtime`` (middleware).
"""

from __future__ import annotations

from datetime import timedelta

from langgraph.config import get_config
from langgraph.prebuilt import ToolRuntime
from langgraph.runtime import Runtime

from ..config import Configuration
from .backend import OpenSandboxBackend


class SandboxFactory:
    """Find or create an OpenSandbox container for the current thread.

    Discovers sandboxes by ``thread_id`` metadata via the OpenSandbox API.
    When no running sandbox is found, creates a new one tagged with the
    current ``thread_id``.
    """

    def __init__(self, runtime: Runtime | ToolRuntime) -> None:
        """Initialize with a runtime context."""
        self._runtime = runtime

    def _get_thread_id(self) -> str:
        """Extract thread_id from runtime config or langgraph context."""
        cfg = getattr(self._runtime, "config", None)
        if not isinstance(cfg, dict):
            cfg = get_config()
        return (cfg.get("configurable") or {}).get("thread_id") or "default"

    def _get_muffin_config(self) -> Configuration:
        """Build a Configuration from runtime config or langgraph context."""
        cfg = getattr(self._runtime, "config", None)
        if not isinstance(cfg, dict):
            cfg = get_config()
        return Configuration.from_runnable_config(cfg)

    def _make_sync_connection(self):
        """Build a ConnectionConfigSync from muffin agent config."""
        from opensandbox.config.connection_sync import ConnectionConfigSync

        config = self._get_muffin_config()
        return ConnectionConfigSync(
            domain=config.opensandbox_url,
            api_key=config.opensandbox_api_key or None,
            protocol="http",
        )

    def _make_async_connection(self):
        """Build a ConnectionConfig (async) from muffin agent config."""
        from opensandbox.config.connection import ConnectionConfig

        config = self._get_muffin_config()
        return ConnectionConfig(
            domain=config.opensandbox_url,
            api_key=config.opensandbox_api_key or None,
            protocol="http",
        )

    def _find_sandbox_id(self) -> str | None:
        """Find a running sandbox tagged with the current thread_id."""
        from opensandbox.models.sandboxes import SandboxFilter
        from opensandbox.sync.manager import SandboxManagerSync

        manager = SandboxManagerSync.create(
            connection_config=self._make_sync_connection(),
        )
        try:
            result = manager.list_sandbox_infos(
                SandboxFilter(
                    states=["Running"],
                    metadata={"thread_id": self._get_thread_id()},
                ),
            )
            return result.sandbox_infos[0].id if result.sandbox_infos else None
        finally:
            manager.close()

    async def _afind_sandbox_id(self) -> str | None:
        """Find a running sandbox tagged with the current thread_id (async)."""
        from opensandbox.manager import SandboxManager
        from opensandbox.models.sandboxes import SandboxFilter

        manager = await SandboxManager.create(
            connection_config=self._make_async_connection(),
        )
        try:
            result = await manager.list_sandbox_infos(
                SandboxFilter(
                    states=["Running"],
                    metadata={"thread_id": self._get_thread_id()},
                ),
            )
            return result.sandbox_infos[0].id if result.sandbox_infos else None
        finally:
            await manager.close()

    def get_sandbox(self):
        """Find or create a sandbox (sync)."""
        from opensandbox.sync.sandbox import SandboxSync

        config = self._get_muffin_config()
        sandbox_id = self._find_sandbox_id()
        if sandbox_id:
            return SandboxSync.connect(
                sandbox_id,
                connection_config=self._make_sync_connection(),
                skip_health_check=False,
            )
        return SandboxSync.create(
            config.opensandbox_image,
            connection_config=self._make_sync_connection(),
            timeout=timedelta(hours=1),
            env={"PYTHONUNBUFFERED": "1"},
            metadata={"thread_id": self._get_thread_id()},
        )

    async def aget_sandbox(self):
        """Find or create a sandbox (async)."""
        from opensandbox.sandbox import Sandbox

        config = self._get_muffin_config()
        sandbox_id = await self._afind_sandbox_id()
        if sandbox_id:
            return await Sandbox.connect(
                sandbox_id,
                connection_config=self._make_async_connection(),
                skip_health_check=False,
            )
        return await Sandbox.create(
            config.opensandbox_image,
            connection_config=self._make_async_connection(),
            timeout=timedelta(hours=1),
            env={"PYTHONUNBUFFERED": "1"},
            metadata={"thread_id": self._get_thread_id()},
        )


def get_backend(runtime: Runtime | ToolRuntime) -> OpenSandboxBackend:
    """BackendFactory: find or create sandbox, return backend (sync).

    Implements the ``BackendFactory`` protocol — pass directly as
    ``backend=get_backend`` to ``create_deep_agent``.
    """
    return OpenSandboxBackend(SandboxFactory(runtime).get_sandbox())


def get_sandbox(runtime: Runtime | ToolRuntime):
    """Find or create a sandbox for the current thread (sync)."""
    return SandboxFactory(runtime).get_sandbox()


async def aget_sandbox(runtime: Runtime | ToolRuntime):
    """Find or create a sandbox for the current thread (async)."""
    return await SandboxFactory(runtime).aget_sandbox()
