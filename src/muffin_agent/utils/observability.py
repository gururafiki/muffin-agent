"""Observability setup for LLM tracing via LangFuse."""

import logging
import os

logger = logging.getLogger(__name__)


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
