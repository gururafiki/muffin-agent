"""OpenSandbox backend for deepagents.

Implements the deepagents BaseSandbox protocol using the OpenSandbox sync SDK,
providing isolated Python execution for financial calculations, dataframe
analysis, and technical indicator computation.
"""

import logging
import threading
from datetime import timedelta

from deepagents.backends.protocol import (
    ExecuteResponse,
    FileDownloadResponse,
    FileUploadResponse,
)
from deepagents.backends.sandbox import BaseSandbox
from opensandbox.sync.sandbox import SandboxSync

from ..config import Configuration


class OpenSandboxBackend(BaseSandbox):
    """deepagents backend backed by an OpenSandbox container.

    Wraps a SandboxSync instance. All operations are synchronous and safe to
    call from a thread (BaseSandbox.aexecute dispatches execute() via
    asyncio.to_thread, so the event loop is never blocked).
    """

    def __init__(self, sandbox: SandboxSync) -> None:
        """Initialize with a connected SandboxSync instance.

        Args:
            sandbox: Connected and ready SandboxSync instance.
        """
        self._sandbox = sandbox

    @property
    def id(self) -> str:
        """Return the unique sandbox container ID."""
        return self._sandbox.id

    # ------------------------------------------------------------------
    # BaseSandbox abstract methods
    # ------------------------------------------------------------------

    def execute(
        self,
        command: str,
        *,
        timeout: int | None = None,
    ) -> ExecuteResponse:
        """Execute a shell command in the sandbox and return the result.

        Args:
            command: Full shell command string to execute.
            timeout: Maximum seconds to wait for completion.

        Returns:
            ExecuteResponse with combined output, exit code, and truncation flag.
        """
        from opensandbox.models.execd import RunCommandOpts

        opts = RunCommandOpts(
            timeout=timedelta(seconds=timeout) if timeout else None,
        )
        execution = self._sandbox.commands.run(command, opts=opts)

        stdout = "".join(m.text for m in execution.logs.stdout)
        stderr = "".join(m.text for m in execution.logs.stderr)
        output = stdout + stderr

        exit_code: int | None = None
        if execution.id:
            try:
                status = self._sandbox.commands.get_command_status(execution.id)
                exit_code = status.exit_code
            except Exception:
                pass  # Non-fatal; exit_code stays None

        if execution.error and exit_code is None:
            exit_code = 1

        return ExecuteResponse(output=output, exit_code=exit_code)

    def upload_files(
        self,
        files: list[tuple[str, bytes]],
    ) -> list[FileUploadResponse]:
        """Upload files to the sandbox filesystem.

        Args:
            files: List of (remote_path, content_bytes) tuples.

        Returns:
            List of FileUploadResponse with per-file success/error status.
        """
        results: list[FileUploadResponse] = []
        for path, content in files:
            try:
                self._sandbox.files.write_file(path, content)
                results.append(FileUploadResponse(path=path))
            except Exception:
                results.append(FileUploadResponse(path=path, error="permission_denied"))
        return results

    def download_files(
        self,
        paths: list[str],
    ) -> list[FileDownloadResponse]:
        """Download files from the sandbox filesystem.

        Args:
            paths: List of remote file paths to download.

        Returns:
            List of FileDownloadResponse with per-file content or error.
        """
        results: list[FileDownloadResponse] = []
        for path in paths:
            try:
                content = self._sandbox.files.read_bytes(path)
                results.append(FileDownloadResponse(path=path, content=content))
            except Exception:
                results.append(FileDownloadResponse(path=path, error="file_not_found"))
        return results

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Release local HTTP resources for this sandbox.

        Does not terminate the remote container — call sandbox.kill() for that.
        """
        self._sandbox.close()


def _make_sync_connection(config: Configuration):
    """Build a ConnectionConfigSync from muffin agent config."""
    from opensandbox.config.connection_sync import ConnectionConfigSync

    return ConnectionConfigSync(
        domain=config.opensandbox_url,
        api_key=config.opensandbox_api_key or None,
        protocol="http",
    )


def _make_async_connection(config: Configuration):
    """Build a ConnectionConfig (async) from muffin agent config."""
    from opensandbox.config.connection import ConnectionConfig

    return ConnectionConfig(
        domain=config.opensandbox_url,
        api_key=config.opensandbox_api_key or None,
        protocol="http",
    )


def create_opensandbox_backend(config: Configuration) -> OpenSandboxBackend:
    """Create a sandbox container and wrap it in an OpenSandboxBackend.

    Blocking — intended for use during agent setup before entering the async
    event loop. Uses SandboxSync so no async bridging is needed.

    Args:
        config: Muffin agent configuration with opensandbox_* fields.

    Returns:
        Connected and ready OpenSandboxBackend.
    """
    sandbox = SandboxSync.create(
        config.opensandbox_image,
        connection_config=_make_sync_connection(config),
        timeout=timedelta(hours=1),
        env={"PYTHONUNBUFFERED": "1"},
    )
    return OpenSandboxBackend(sandbox)


async def create_opensandbox_sandbox(config: Configuration):
    """Create an async Sandbox for use in async tools.

    Uses the native async OpenSandbox SDK — no sync bridging. Intended for
    tools that run inside an already-running async event loop.

    Args:
        config: Muffin agent configuration with opensandbox_* fields.

    Returns:
        Connected and ready async Sandbox instance.
    """
    from opensandbox.sandbox import Sandbox

    return await Sandbox.create(
        config.opensandbox_image,
        connection_config=_make_async_connection(config),
        timeout=timedelta(hours=1),
        env={"PYTHONUNBUFFERED": "1"},
    )


_log = logging.getLogger(__name__)


class SandboxFactory:
    """Per-thread_id sandbox factory that implements BackendFactory.

    Pass an instance as ``backend=`` to ``create_deep_agent``. deepagents
    middleware calls the instance on every tool invocation; the factory
    reconnects to (or provisions) the sandbox for that ``thread_id`` so all
    tool calls within a conversation share one container.

    Only sandbox IDs are stored in memory — no live connection objects are
    held between calls. ``SandboxSync.connect()`` is used on every invocation
    (with ``skip_health_check=True`` to avoid blocking), so the overhead is
    one lightweight client construction per tool call rather than a round-trip
    health check.

    Usage::

        factory = SandboxFactory(config)
        agent = create_deep_agent(model=llm, backend=factory, ...)
    """

    def __init__(self, config: Configuration) -> None:
        """Initialize with muffin agent configuration."""
        self._config = config
        self._sandbox_ids: dict[str, str] = {}  # thread_id → sandbox_id
        self._lock = threading.Lock()

    def __call__(self, runtime) -> OpenSandboxBackend:
        """Return a backend for the current thread_id, creating one if needed.

        Called by deepagents middleware on every tool invocation.
        """
        thread_id: str = (
            (runtime.config.get("configurable") or {}).get("thread_id") or "default"
        )

        with self._lock:
            existing_id = self._sandbox_ids.get(thread_id)

        if existing_id:
            try:
                sandbox = SandboxSync.connect(
                    existing_id,
                    connection_config=_make_sync_connection(self._config),
                    skip_health_check=True,
                )
                return OpenSandboxBackend(sandbox)
            except Exception:
                _log.info(
                    "Sandbox %s unreachable for thread %s, creating new one",
                    existing_id,
                    thread_id,
                )

        backend = create_opensandbox_backend(self._config)
        _log.info("Created sandbox %s for thread %s", backend.id, thread_id)
        with self._lock:
            self._sandbox_ids[thread_id] = backend.id
        return backend
