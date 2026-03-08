"""Local sandbox backend for Python code execution.

Runs shell commands as subprocesses in the local Python environment,
giving the stock evaluation agent access to TA-Lib, pandas, and any
other packages installed in the current virtualenv.
"""

import os
import subprocess
import tempfile
import uuid

from deepagents.backends.protocol import (
    ExecuteResponse,
    FileDownloadResponse,
    FileUploadResponse,
)
from deepagents.backends.sandbox import BaseSandbox

_DEFAULT_TIMEOUT = 120


class LocalSandbox(BaseSandbox):
    """Execute shell commands in a local subprocess.

    Suitable for development environments where TA-Lib, pandas, and
    other native packages are installed alongside the agent. All commands
    run inside an isolated working directory; no sandboxing beyond that.

    For production or untrusted code use a cloud sandbox (Daytona, Runloop,
    Modal) by swapping this backend in ``Configuration.get_sandbox()``.
    """

    def __init__(
        self,
        work_dir: str | None = None,
        timeout: int = _DEFAULT_TIMEOUT,
    ) -> None:
        """Initialize the local sandbox.

        Args:
            work_dir: Working directory for command execution and file I/O.
                Defaults to a fresh temporary directory per sandbox instance.
            timeout: Default command timeout in seconds.
        """
        self._work_dir = work_dir or tempfile.mkdtemp(prefix="muffin_sandbox_")
        self._timeout = timeout
        self._id = f"local-{uuid.uuid4().hex[:8]}"

    @property
    def id(self) -> str:
        """Return the unique sandbox identifier."""
        return self._id

    def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
        """Run a shell command, capturing stdout and stderr.

        Args:
            command: Shell command string to execute.
            timeout: Per-call timeout override in seconds.

        Returns:
            ExecuteResponse with combined stdout/stderr and exit code.
        """
        effective_timeout = timeout if timeout is not None else self._timeout
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                cwd=self._work_dir,
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
        """Write files into the sandbox working directory.

        Args:
            files: Sequence of (path, content) pairs. Paths are relative
                to work_dir; leading slashes are stripped.

        Returns:
            One FileUploadResponse per file, with error set on failure.
        """
        responses = []
        for path, content in files:
            dest = os.path.join(self._work_dir, path.lstrip("/"))
            os.makedirs(os.path.dirname(dest) or self._work_dir, exist_ok=True)
            try:
                with open(dest, "wb") as f:
                    f.write(content)
                responses.append(FileUploadResponse(path=path))
            except OSError:
                responses.append(FileUploadResponse(path=path, error="permission_denied"))
        return responses

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Read files from the sandbox working directory.

        Args:
            paths: File paths relative to work_dir; leading slashes stripped.

        Returns:
            One FileDownloadResponse per path, with error set on failure.
        """
        responses = []
        for path in paths:
            src = os.path.join(self._work_dir, path.lstrip("/"))
            try:
                with open(src, "rb") as f:
                    responses.append(FileDownloadResponse(path=path, content=f.read()))
            except FileNotFoundError:
                responses.append(FileDownloadResponse(path=path, error="file_not_found"))
            except IsADirectoryError:
                responses.append(FileDownloadResponse(path=path, error="is_directory"))
        return responses
