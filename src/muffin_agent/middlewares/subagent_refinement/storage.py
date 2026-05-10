"""Filesystem-backed storage for subagent findings.

Uses the agent's ``/scratch/`` route (thread-scoped state backend) so
findings persist for the duration of the parent's run but are scoped to
the thread — no cross-thread contamination, no need for a persistent
store.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence

from deepagents.backends.protocol import BackendProtocol
from langchain_core.messages import AnyMessage, HumanMessage

from .schema import CollectionFindings

logger = logging.getLogger(__name__)

_SCRATCH_DIR = "/scratch/subagent_runs"
# Require the marker to end at a word/string boundary so unsafe trailing
# characters (slashes, dots) reject the id outright instead of being
# silently truncated.
_CALL_ID_RE = re.compile(r"prior_call_id=([A-Za-z0-9_-]+)(?=\s|$)")
_SAFE_CALL_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def call_id_path(call_id: str) -> str:
    """Return the canonical scratch path for *call_id*'s findings file."""
    if not _SAFE_CALL_ID.match(call_id):
        raise ValueError(f"Unsafe call_id: {call_id!r}")
    return f"{_SCRATCH_DIR}/{call_id}.json"


def extract_prior_call_id(text: str) -> str | None:
    """Pull a ``prior_call_id=<id>`` marker out of a task description."""
    if not isinstance(text, str):
        return None
    match = _CALL_ID_RE.search(text)
    if not match:
        return None
    candidate = match.group(1)
    return candidate if _SAFE_CALL_ID.match(candidate) else None


async def read_findings(
    backend: BackendProtocol, call_id: str
) -> CollectionFindings | None:
    """Load a prior findings file, returning ``None`` on any failure."""
    path = call_id_path(call_id)
    try:
        result = await backend.aread(path)
    except Exception:
        logger.debug("Findings read failed for %s", call_id, exc_info=True)
        return None
    if result.error is not None or result.file_data is None:
        return None
    raw = result.file_data.get("content")
    if not isinstance(raw, str) or not raw.strip():
        return None
    try:
        return CollectionFindings.model_validate_json(raw)
    except Exception:
        logger.debug("Findings parse failed for %s", call_id, exc_info=True)
        return None


async def write_findings(
    backend: BackendProtocol, findings: CollectionFindings
) -> None:
    """Persist findings under their ``call_id``, swallowing backend errors."""
    path = call_id_path(findings.call_id)
    payload = findings.model_dump_json()
    try:
        await backend.awrite(path, payload)
    except Exception:
        logger.debug("Findings write failed for %s", findings.call_id, exc_info=True)


def latest_human_text(messages: Sequence[AnyMessage]) -> str:
    """Return the most recent ``HumanMessage`` content as a string."""
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            content = message.content
            return content if isinstance(content, str) else str(content)
    return ""
