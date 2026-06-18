"""The sandbox seam — in-memory backend + ``execute_python`` stand-in.

Two boundaries are faked together:

1. **The default backend** (``get_backend``). Agents built with
   ``.with_sandbox()`` carry deepagents' ``FilesystemMiddleware`` with a
   sandbox-capable default route. The middleware *resolves that backend on every
   model call* (to decide whether to expose the built-in ``execute`` tool), so the
   backend is NOT lazy for these agents — without a patch it dials OpenSandbox and
   raises ``ConnectError``. We swap it for an in-memory ``StateBackend`` (no
   network; execution unsupported, so the unused ``execute`` tool is simply
   filtered out).
2. **``execute_python``** (``aget_sandbox``). Our own ``execute_python`` tool
   drives a sandbox as an async context manager: ``files.write_file`` then
   ``commands.run`` (result exposes ``.logs.stdout`` / ``.logs.stderr`` of objects
   with ``.text``, plus ``.id`` / ``.error``) and ``commands.get_command_status``.
   The fake reproduces just that surface and returns a canned stdout.

Wrap ``patch_sandbox()`` for **any agent that calls ``.with_sandbox()``**. Plain
ReAct/deep agents without a sandbox never resolve the backend and need no patch.
"""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
from types import SimpleNamespace
from typing import Iterator
from unittest.mock import patch


class _FakeSandbox:
    """Minimal async ``execute_python`` sandbox returning a canned stdout."""

    def __init__(self, stdout: str = "(no output)", *, exit_code: int = 0) -> None:
        self._stdout = stdout
        self._exit_code = exit_code

    async def __aenter__(self) -> "_FakeSandbox":
        return self

    async def __aexit__(self, *exc: object) -> bool:
        return False

    # files.* — execute_python writes the code file then deletes it
    @property
    def files(self) -> "_FakeSandbox":
        return self

    async def write_file(self, path: str, content: str) -> None:  # noqa: ARG002
        return None

    # commands.* — run the file, then query exit status
    @property
    def commands(self) -> "_FakeSandbox":
        return self

    async def run(self, command: str):  # noqa: ANN201, ARG002
        return SimpleNamespace(
            id="fake-cmd",
            error=None,
            logs=SimpleNamespace(
                stdout=[SimpleNamespace(text=self._stdout)],
                stderr=[],
            ),
        )

    async def get_command_status(self, command_id: str):  # noqa: ANN201, ARG002
        return SimpleNamespace(exit_code=self._exit_code)


def _in_memory_backend(_runtime: object):  # noqa: ANN202
    """A ``BackendFactory`` returning an in-memory ``StateBackend`` (no network)."""
    from deepagents.backends import StateBackend

    return StateBackend()


# get_backend is imported into several namespaces; patch every binding that an
# agent factory might resolve so this helper works regardless of agent type.
_GET_BACKEND_TARGETS = (
    "muffin_agent.utils.agent_builder.get_backend",  # MuffinAgentBuilder.with_sandbox
    "muffin_agent.sandbox.factory.get_backend",  # canonical source
    "muffin_agent.sandbox.get_backend",  # package re-export
)


@contextmanager
def patch_sandbox(
    execute_output: str = "(no output)", *, exit_code: int = 0
) -> Iterator[_FakeSandbox]:
    """Patch the sandbox seams: in-memory ``get_backend`` + fake ``aget_sandbox``.

    *execute_output* is the stdout ``execute_python`` will return.
    """
    sandbox = _FakeSandbox(execute_output, exit_code=exit_code)

    async def _aget_sandbox(_runtime: object) -> _FakeSandbox:
        return sandbox

    with ExitStack() as stack:
        for target in _GET_BACKEND_TARGETS:
            try:
                stack.enter_context(patch(target, new=_in_memory_backend))
            except (AttributeError, ModuleNotFoundError):
                pass  # binding not present in this import graph — skip
        stack.enter_context(
            patch("muffin_agent.sandbox.tools.aget_sandbox", new=_aget_sandbox)
        )
        yield sandbox
