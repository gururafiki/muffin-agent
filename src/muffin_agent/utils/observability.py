"""Observability setup for LLM tracing via LangFuse.

Two entry points, one per invocation model:

* ``setup_tracing()`` — the **CLI path**. The caller owns the invocation, so it
  attaches the returned handler(s) per-run via ``RunnableConfig(callbacks=...)``.
* ``instrument_graph()`` — the **LangGraph Platform / server path**. The server
  owns the invocation, so we bake the handler into the compiled graph with
  ``with_config`` — the pattern LangFuse documents for LangGraph Server. This is
  what makes tracing work for the deployed graphs in ``langgraph.json`` (the CLI
  callbacks never reach them).

Credentials are read from the environment by the LangFuse client itself:
``LANGFUSE_PUBLIC_KEY`` / ``LANGFUSE_SECRET_KEY`` / ``LANGFUSE_BASE_URL`` (the
v4 name; ``LANGFUSE_HOST`` is the deprecated fallback).
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph

logger = logging.getLogger(__name__)

_DEFAULT_HOST = "https://cloud.langfuse.com"


def _langfuse_configured() -> bool:
    """Return whether LangFuse credentials are present in the environment."""
    return bool(
        os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY")
    )


def _build_callback_handler():
    """Build a LangFuse LangChain ``CallbackHandler``, or ``None`` if unavailable.

    Env-gated and non-blocking: unlike ``setup_tracing`` this never calls
    ``client.auth_check()`` (LangFuse documents it as blocking and discourages it
    in production / hot paths), so it is safe to invoke at import / server start.
    Returns ``None`` when credentials are absent or the package is not installed,
    so callers degrade gracefully.
    """
    if not _langfuse_configured():
        logger.debug(
            "LANGFUSE_PUBLIC_KEY/LANGFUSE_SECRET_KEY not set — tracing disabled"
        )
        return None

    try:
        from langfuse.langchain import CallbackHandler
    except ImportError:
        logger.error(
            "langfuse not installed — tracing disabled. "
            "Install with: pip install langfuse"
        )
        return None

    logger.info(
        "LangFuse tracing enabled (host=%s)",
        os.environ.get("LANGFUSE_BASE_URL", _DEFAULT_HOST),
    )
    return CallbackHandler()


def instrument_graph(graph: CompiledStateGraph) -> CompiledStateGraph:
    """Bake LangFuse tracing into a compiled graph for server-owned invocation.

    On LangGraph Platform the server owns the invocation, so per-run
    ``RunnableConfig(callbacks=...)`` wiring (as the CLI does) never reaches the
    deployed graphs. Instead we attach the handler once via ``with_config`` — the
    integration LangFuse documents for LangGraph Server. ``Pregel.with_config``
    returns ``Self``, so the result is still a ``CompiledStateGraph`` and Platform
    autodiscovery / schemas are unaffected.

    Returns the graph unchanged when LangFuse is unavailable (no credentials, or
    the package is not installed), so import never fails on account of tracing.

    Args:
        graph: The compiled LangGraph to trace.

    Returns:
        The same graph with a LangFuse callback attached, or unchanged.
    """
    handler = _build_callback_handler()
    if handler is None:
        return graph
    return graph.with_config({"callbacks": [handler]})


def setup_tracing(*, session_id: str | None = None) -> list:
    """Initialize LangFuse tracing and return callback handlers.

    Reads credentials from environment variables:
        LANGFUSE_SECRET_KEY, LANGFUSE_PUBLIC_KEY, LANGFUSE_BASE_URL

    Args:
        session_id: Optional session ID for grouping traces (e.g. ticker).
            With langfuse v4, session_id propagation requires a
            ``propagate_attributes()`` context manager wrapped around the
            agent invocation — see the langfuse v3 → v4 upgrade guide.
            If supplied here it's logged but not currently wired; callers
            who need session-scoped tracing should use
            ``langfuse.propagate_attributes`` directly until we expose
            a per-invocation context helper.

    Returns:
        List of callback handlers to pass to LangGraph config.
        Empty list if LangFuse is unavailable.
    """
    try:
        from langfuse import get_client
        from langfuse.langchain import CallbackHandler

        client = get_client()
        client.auth_check()

        handler = CallbackHandler()

        if session_id:
            logger.debug(
                "session_id=%s requested but langfuse v4 requires "
                "propagate_attributes() context — skipping",
                session_id,
            )

        logger.info(
            "LangFuse tracing enabled (host=%s)",
            os.environ.get("LANGFUSE_BASE_URL", "cloud.langfuse.com"),
        )
        return [handler]

    except ImportError:
        logger.error(
            "langfuse not installed — tracing disabled. "
            "Install with: pip install langfuse"
        )
        return []
    except Exception as exc:
        logger.warning("LangFuse tracing unavailable: %s", exc)
        return []
