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
from .memory_config import (
    MEMORIES_NAMESPACE_ROOT,
    MemoryConfiguration,
    MemoryUnavailableError,
)

# Re-export ``MemoryUnavailableError`` for backwards compatibility — its
# canonical home is now ``utils.memory_config`` (alongside ``MemoryConfiguration``).
__all__ = [
    "MemoryUnavailableError",
    "get_agent_backend",
    "make_agent_backend",
]

_SKILLS_ROOT = Path(__file__).resolve().parents[1] / "skills"


def _memories_namespace(_rt: Runtime[Any]) -> tuple[str, ...]:
    """Return the StoreBackend namespace for long-term memory.

    Thin wrapper over :meth:`MemoryConfiguration.resolve_user_id` — see
    that method for the full resolution chain.  When called outside a
    LangGraph invocation context (no ``get_config()`` available) the
    chain falls through to the env-only fallback path on an empty
    configurable dict.
    """
    try:
        config: RunnableConfig = get_config()
    except (RuntimeError, KeyError):
        # Outside a graph execution (ad-hoc test, deferred task without
        # context) — fall through to env-only fallback by passing an
        # empty config to resolve_user_id.
        config = cast(RunnableConfig, {})
    user_id = MemoryConfiguration.resolve_user_id(config, allow_missing=False)
    return (*MEMORIES_NAMESPACE_ROOT, user_id)


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
