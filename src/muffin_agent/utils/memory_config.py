"""Configuration for the ``/memories/`` persistent-memory route."""

from pydantic import Field

from .base_config import BaseConfiguration


class MemoryConfiguration(BaseConfiguration):
    """Runtime configuration consulted by ``_memories_namespace``.

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
