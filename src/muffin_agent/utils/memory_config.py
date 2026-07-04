"""Configuration for the ``/memories/`` persistent-memory route."""

from typing import Literal, overload

from langchain_core.runnables import RunnableConfig
from pydantic import Field

from .base_config import BaseConfiguration

MEMORIES_NAMESPACE_ROOT: tuple[str, ...] = ("memories",)
"""Shared namespace root prefix for per-user long-term memory.

Used by :func:`muffin_agent.utils.backends._memories_namespace` to compose
the ``("memories", user_id)`` tuple for the ``/memories/`` filesystem route,
and by :class:`muffin_agent.agents.trading_decision.reflection.memory.ReflectionMemory`
as the prefix of its decision-records namespace ``("memories", user_id, "decisions")``.
"""


_NON_USER_IDENTITIES = frozenset({"anonymous", "api-client"})
"""Auth-hook sentinel identities that are not real users.

``auth.py`` returns ``anonymous`` (auth disabled / optional sign-in without a
credential) and ``api-client`` (shared bearer token). Neither should become a
memory namespace — anonymous traffic keeps flowing through the existing
``user_id`` / ``memory_debug_user_id`` chain instead.
"""


class MemoryUnavailableError(LookupError):
    """Raised when a per-user memory namespace cannot be resolved.

    Inherits from :class:`LookupError` so callers that can tolerate missing
    memory (the optional memory-load middleware, tool-error handler) can
    catch a narrow type; generic ``LookupError`` / ``Exception`` handlers
    propagate it as a normal error the LLM can see through the tool-error
    channel.
    """


class MemoryConfiguration(BaseConfiguration):
    """Runtime configuration for per-user memory resolution.

    Fields follow the standard :class:`BaseConfiguration` pattern: each
    field is populated from the env var matching ``field_name.upper()``
    first, then from ``RunnableConfig['configurable'][field_name]``.
    """

    memory_debug_user_id: str | None = Field(
        default=None,
        description=(
            "Debug-only fallback user_id used when "
            "RunnableConfig['configurable'] lacks 'user_id'.  Read from env "
            "MEMORY_DEBUG_USER_ID or configurable.memory_debug_user_id.  "
            "Intended for local debugging with clients that do not populate "
            "configurable.user_id (e.g. agent-chat-ui).  Do NOT set in "
            "multi-user deployments — it collapses all anonymous traffic "
            "onto a single shared namespace."
        ),
    )

    @overload
    @classmethod
    def resolve_user_id(
        cls,
        config: RunnableConfig,
        *,
        allow_missing: Literal[False] = False,
    ) -> str: ...

    @overload
    @classmethod
    def resolve_user_id(
        cls,
        config: RunnableConfig,
        *,
        allow_missing: Literal[True],
    ) -> str | None: ...

    @classmethod
    def resolve_user_id(
        cls,
        config: RunnableConfig,
        *,
        allow_missing: bool = False,
    ) -> str | None:
        """Resolve ``user_id`` from a ``RunnableConfig`` with layered fallback.

        Resolution order:

        1. ``configurable.langgraph_auth_user_id`` — the identity VERIFIED by
           the LangGraph auth hook (``auth.py``) and injected server-side into
           every run's configurable. Tamper-proof: a client cannot spoof
           another user's memory namespace by sending a different ``user_id``.
           The non-user sentinels (``anonymous`` / ``api-client``) are skipped.
        2. ``configurable.user_id`` — client-supplied identity (CLI ``--user``,
           anonymous app users, local dev without auth).
        3. :attr:`memory_debug_user_id` (env ``MEMORY_DEBUG_USER_ID`` or
           ``configurable.memory_debug_user_id``) — debug-only fallback.
        4. If *allow_missing* is ``True``, return ``None``; otherwise raise
           :class:`MemoryUnavailableError`.

        This is the single source of truth for ``user_id`` resolution across
        the codebase. ``_memories_namespace`` (in ``utils/backends.py``) uses
        ``allow_missing=False`` so missing user_id surfaces as an error; the
        reflection bookend nodes use ``allow_missing=True`` so the trading
        pipeline degrades gracefully when memory infrastructure isn't wired.
        """
        configurable = dict(config.get("configurable") or {})
        verified = configurable.get("langgraph_auth_user_id")
        if (
            isinstance(verified, str)
            and verified
            and verified not in _NON_USER_IDENTITIES
        ):
            return verified
        user_id = configurable.get("user_id")
        if isinstance(user_id, str) and user_id:
            return user_id
        try:
            debug = cls.from_runnable_config(config).memory_debug_user_id
        except Exception:
            debug = None
        if isinstance(debug, str) and debug:
            return debug
        if allow_missing:
            return None
        raise MemoryUnavailableError(
            "Persistent memory is disabled for this request: "
            "configurable.user_id is not set. CLI: pass --user. "
            "Server: set configurable.user_id per request, or set "
            "MEMORY_DEBUG_USER_ID for local debugging."
        )
