"""Docker sandbox backend for isolated Python code execution.

Provides real process, filesystem, and network isolation via Docker containers.
Each DockerSandbox maps to a single long-running container started on ``__init__``
and removed on ``close()`` or garbage collection.

Security controls applied by default:
- ``--network=none`` — no inbound or outbound network access
- ``--cap-drop=ALL`` — all Linux capabilities dropped
- ``--security-opt=no-new-privileges`` — no privilege escalation
- ``--memory`` / ``--cpus`` — resource quotas prevent DoS

For TA-Lib support, build a custom image::

    FROM python:3.12-slim
    RUN apt-get update && apt-get install -y --no-install-recommends \\
            gcc libta-lib-dev \\
        && pip install --no-cache-dir ta-lib pandas numpy \\
        && apt-get purge -y gcc && rm -rf /var/lib/apt/lists/*
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import uuid
from pathlib import Path

from deepagents.backends.protocol import (
    ExecuteResponse,
    FileDownloadResponse,
    FileUploadResponse,
)
from deepagents.backends.sandbox import BaseSandbox

_DEFAULT_TIMEOUT = 120
_DEFAULT_IMAGE = "python:3.12-slim"
_DEFAULT_WORK_DIR = "/workspace"


class DockerSandbox(BaseSandbox):
    """Execute commands inside an isolated Docker container.

    Each instance owns one container for its lifetime. All ``execute()`` calls
    run inside that container via ``docker exec``. File I/O uses ``docker cp``.

    Call ``close()`` explicitly when done, or use as a context manager::

        with DockerSandbox(image="my-talib-image") as sandbox:
            config = Configuration(sandbox_backend="docker")
            ...
    """

    def __init__(
        self,
        image: str = _DEFAULT_IMAGE,
        work_dir: str = _DEFAULT_WORK_DIR,
        timeout: int = _DEFAULT_TIMEOUT,
        network: str = "none",
        memory: str = "512m",
        cpus: float = 1.0,
        env: dict[str, str] | None = None,
    ) -> None:
        """Start a Docker container for sandboxed execution.

        Args:
            image: Docker image. Must have Python and any required packages
                (TA-Lib, pandas, etc.).
            work_dir: Absolute working directory inside the container.
            timeout: Default command timeout in seconds.
            network: Docker network mode. ``"none"`` (default) disables all
                networking for maximum isolation.
            memory: Container memory limit, e.g. ``"512m"`` or ``"2g"``.
            cpus: CPU quota. ``1.0`` = one logical CPU.
            env: Additional environment variables to inject into the container.

        Raises:
            RuntimeError: If Docker is not installed or the container fails to start.
        """
        self._work_dir = work_dir
        self._timeout = timeout
        self._sandbox_id = f"docker-{uuid.uuid4().hex[:8]}"
        self._container_id: str | None = None

        cmd = [
            "docker", "run",
            "--detach",
            "--rm",
            f"--name={self._sandbox_id}",
            f"--network={network}",
            f"--memory={memory}",
            f"--cpus={cpus}",
            f"--workdir={work_dir}",
            "--security-opt=no-new-privileges",
            "--cap-drop=ALL",
        ]
        for k, v in (env or {}).items():
            cmd.extend(["--env", f"{k}={v}"])
        cmd.extend([image, "sleep", "infinity"])

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"Failed to start Docker container: {exc.stderr.strip()}"
            ) from exc
        except FileNotFoundError as exc:
            raise RuntimeError(
                "Docker CLI not found. Install Docker: https://docs.docker.com/get-docker/"
            ) from exc

        self._container_id = result.stdout.strip()

        # Ensure work dir exists (the image may not pre-create it).
        subprocess.run(
            ["docker", "exec", self._container_id, "mkdir", "-p", work_dir],
            capture_output=True,
        )

    @property
    def id(self) -> str:
        """Return the unique sandbox identifier."""
        return self._sandbox_id

    def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
        """Run a shell command inside the container.

        Args:
            command: Shell command string to execute.
            timeout: Per-call timeout override in seconds.

        Returns:
            ExecuteResponse with combined stdout/stderr and exit code.
        """
        if not self._container_id:
            return ExecuteResponse(output="Sandbox is closed.", exit_code=1)

        effective_timeout = timeout if timeout is not None else self._timeout
        try:
            result = subprocess.run(
                [
                    "docker", "exec",
                    f"--workdir={self._work_dir}",
                    self._container_id,
                    "sh", "-c", command,
                ],
                capture_output=True,
                text=True,
                timeout=effective_timeout,
            )
            return ExecuteResponse(
                output=result.stdout + result.stderr,
                exit_code=result.returncode,
            )
        except subprocess.TimeoutExpired:
            return ExecuteResponse(
                output=f"Command timed out after {effective_timeout}s",
                exit_code=124,
            )

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """Copy files into the container working directory.

        Args:
            files: Sequence of (path, content) pairs. Paths are relative to
                work_dir; leading slashes are stripped.

        Returns:
            One FileUploadResponse per file, with error set on failure.
        """
        responses = []
        for path, content in files:
            dest = f"{self._work_dir}/{path.lstrip('/')}"
            parent = str(Path(dest).parent)
            tmp_fd, tmp_path = tempfile.mkstemp()
            try:
                os.write(tmp_fd, content)
                os.close(tmp_fd)
                subprocess.run(
                    ["docker", "exec", self._container_id, "mkdir", "-p", parent],
                    capture_output=True,
                )
                result = subprocess.run(
                    ["docker", "cp", tmp_path, f"{self._container_id}:{dest}"],
                    capture_output=True,
                )
                if result.returncode == 0:
                    responses.append(FileUploadResponse(path=path))
                else:
                    responses.append(FileUploadResponse(path=path, error="permission_denied"))
            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
        return responses

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Copy files out of the container working directory.

        Args:
            paths: File paths relative to work_dir; leading slashes stripped.

        Returns:
            One FileDownloadResponse per path, with error set on failure.
        """
        responses = []
        for path in paths:
            src = f"{self._work_dir}/{path.lstrip('/')}"
            tmp_fd, tmp_path = tempfile.mkstemp()
            os.close(tmp_fd)
            try:
                result = subprocess.run(
                    ["docker", "cp", f"{self._container_id}:{src}", tmp_path],
                    capture_output=True,
                    text=True,
                )
                if result.returncode == 0:
                    with open(tmp_path, "rb") as f:
                        responses.append(FileDownloadResponse(path=path, content=f.read()))
                elif "No such file" in result.stderr:
                    responses.append(FileDownloadResponse(path=path, error="file_not_found"))
                else:
                    responses.append(FileDownloadResponse(path=path, error="permission_denied"))
            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
        return responses

    def close(self) -> None:
        """Stop and remove the container.

        Safe to call multiple times. The container is started with ``--rm``
        so Docker removes it automatically once stopped.
        """
        if self._container_id:
            subprocess.run(
                ["docker", "stop", "--time=5", self._container_id],
                capture_output=True,
            )
            self._container_id = None

    def __enter__(self) -> "DockerSandbox":
        """Support use as a context manager."""
        return self

    def __exit__(self, *_: object) -> None:
        """Stop the container on context exit."""
        self.close()

    def __del__(self) -> None:
        self.close()
