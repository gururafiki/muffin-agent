"""Reusable backend primitives for skills-enabled and memory-enabled agents."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from deepagents.backends import CompositeBackend, FilesystemBackend
from deepagents.backends.state import StateBackend
from deepagents.backends.store import StoreBackend
from langchain_core.runnables import RunnableConfig
from langgraph.config import get_config
from langgraph.runtime import Runtime

from ..sandbox import get_backend
from .memory_config import MemoryConfiguration

_SKILLS_ROOT = Path(__file__).resolve().parents[1] / "skills"
_MEMORIES_NAMESPACE_ROOT = "memories"


class MemoryUnavailableError(LookupError):
    """Raised by ``_memories_namespace`` when ``configurable.user_id`` is absent.

    Callers that can tolerate missing memory (the optional memory-load
    middleware, tool-error handler) catch this specifically.  Generic
    ``LookupError``/``Exception`` handlers propagate it as a normal error
    the LLM can see through the tool-error channel.
    """


def _memories_namespace(_rt: Runtime[Any]) -> tuple[str, ...]:
    """Return the StoreBackend namespace for long-term memory.

    Resolution order:

    1. ``configurable.user_id`` from
       :func:`langgraph.config.get_config` — the real per-request
       identity, set by the CLI (``--user``) or the server.
    2. :class:`MemoryConfiguration.memory_debug_user_id` (env
       ``MEMORY_DEBUG_USER_ID`` or ``configurable.memory_debug_user_id``)
       — debug-only fallback for local development against clients that
       do not populate ``user_id`` (e.g. agent-chat-ui).  Collapses all
       anonymous traffic onto a single shared namespace; do NOT set in
       multi-user deployments.
    3. Otherwise raise :class:`MemoryUnavailableError`.

    The argument is a :class:`langgraph.runtime.Runtime` (or the
    deprecation shim ``_NamespaceRuntimeCompat`` from deepagents 0.5.x)
    but is intentionally unused — config is fetched canonically via
    :func:`get_config`, matching deepagents' own ``StoreBackend``
    helpers.  The parameter is retained to satisfy ``NamespaceFactory``
    and to leave room for future ``rt.context``-based resolution.
    """
    try:
        config: RunnableConfig = get_config()
    except (RuntimeError, KeyError):
        # Outside a graph execution (ad-hoc test, deferred task without
        # context) — fall through to env-only fallback.
        config = cast(RunnableConfig, {})
    configurable = config.get("configurable") or {}
    user_id = configurable.get("user_id")
    if isinstance(user_id, str) and user_id:
        return (_MEMORIES_NAMESPACE_ROOT, user_id)
    debug_id = MemoryConfiguration.from_runnable_config(config).memory_debug_user_id
    if debug_id:
        return (_MEMORIES_NAMESPACE_ROOT, debug_id)
    raise MemoryUnavailableError(
        "Persistent memory is disabled for this request: "
        "configurable.user_id is not set. CLI: pass --user. "
        "Server: set configurable.user_id per request, or set "
        "MEMORY_DEBUG_USER_ID for local debugging."
    )


def make_agent_backend(skills_root: Path | str = _SKILLS_ROOT):
    """Build a ``BackendFactory`` routing four paths across three backends.

    Retained as a convenience for tests and callers that want a composite
    backend without the full ``MuffinAgentBuilder``.  Routes:

    - default → ``OpenSandboxBackend`` (via :func:`get_backend`)
    - ``/skills/`` → ``FilesystemBackend(root_dir=skills_root)``
    - ``/scratch/`` → ``StateBackend``
    - ``/memories/`` → ``StoreBackend(namespace=_memories_namespace)``
    """
    skills_path = Path(skills_root)

    def factory(runtime: Any) -> CompositeBackend:
        return CompositeBackend(
            default=get_backend(runtime),
            routes={
                "/skills/": FilesystemBackend(root_dir=skills_path, virtual_mode=True),
                "/scratch/": StateBackend(),
                "/memories/": StoreBackend(namespace=_memories_namespace),
            },
        )

    return factory


get_agent_backend = make_agent_backend()
"""Default backend factory for muffin agents (skills_root=_SKILLS_ROOT)."""
