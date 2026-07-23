"""Offload heavy per-subagent detail (transcript + tools + output) to the Store.

Layout mirrors the tool-result cache: namespace ``("subagent_detail", thread_id)``,
key = the tree node id, value = ``{messages, tool_runs, output}``. Best-effort —
store failures are swallowed so capture never breaks a run.
"""

from __future__ import annotations

import logging
from typing import Any

from langgraph.store.base import BaseStore

logger = logging.getLogger(__name__)


async def offload_subagent_detail(
    store: BaseStore | None,
    thread_id: str,
    node_id: str,
    *,
    messages: list[dict[str, Any]],
    tool_runs: list[dict[str, Any]],
    output: Any,
) -> bool:
    """Write a sub-agent tree node's heavy detail to the Store.

    Args:
        store: The shared ``BaseStore``, or ``None`` to skip (no-op).
        thread_id: The run's thread id (namespace scoping).
        node_id: The tree node id this detail belongs to (see ``tree.py``).
        messages: Serialized transcript (``serialize_messages`` output).
        tool_runs: Per-tool execution records (``build_tool_records`` output).
        output: The agent's final output (structured response or free-form).

    Returns:
        ``True`` on successful write; ``False`` when ``store`` is ``None`` or
        the write fails.
    """
    if store is None:
        return False
    try:
        await store.aput(
            ("subagent_detail", thread_id),
            node_id,
            {"messages": messages, "tool_runs": tool_runs, "output": output},
        )
    except Exception:
        logger.debug(
            "offload_subagent_detail failed for %s/%s",
            thread_id,
            node_id,
            exc_info=True,
        )
        return False
    return True
