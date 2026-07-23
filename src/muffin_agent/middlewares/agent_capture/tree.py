"""Execution-topology records for the sub-agent tree (light; rides thread.values).

Complements ``records.py`` (per-tool-call records): where ``records.py`` answers
"what tools did this agent call", this module answers "who called whom" — one
``TreeNode`` per capturing agent, keyed by a stable id derived from LangGraph's
``checkpoint_ns``. Nodes accumulate across nested/parallel captures via
``merge_subagent_tree``, a reducer suitable for an ``operator``-annotated state
channel.
"""

from __future__ import annotations

import uuid
from typing import Any, Literal, TypedDict

_ROOT = "__root__"


class ToolSummary(TypedDict):
    """Aggregate counts of a captured agent's tool calls."""

    count: int
    tools: list[str]
    ok: int
    failed: int
    cached: int


class TreeNode(TypedDict):
    """One node in the sub-agent execution tree."""

    id: str
    parent_id: str | None
    name: str
    kind: Literal["subgraph", "task"]
    status: Literal["ok", "error"]
    tool_summary: ToolSummary
    output_preview: str | None
    has_detail: bool


def _strip_capture_segments(segments: list[str]) -> list[str]:
    """Drop trailing capturing-middleware segments from a ns segment list.

    The capturing middleware's OWN node is always the last ``checkpoint_ns``
    segment (validated in the Task-1 spike) — e.g.
    ``AgentCaptureMiddleware.after_agent:<uuid>``. Strip it (and any further
    trailing occurrences) so the remaining segments identify the captured
    agent itself.
    """
    while segments and segments[-1].split(":", 1)[0].startswith(
        "AgentCaptureMiddleware"
    ):
        segments = segments[:-1]
    return segments


def node_ids_from_ns(checkpoint_ns: str | None) -> tuple[str, str | None]:
    """Map a LangGraph ``checkpoint_ns`` to ``(id, parent_id)``.

    Namespaces are ``|``-joined ``<node>:<task_id>`` segments; the LAST segment
    is the capturing middleware's own node and is stripped. The id is then the
    cleaned ns; the parent is the cleaned ns minus its last segment
    (``__root__`` at depth 1). Ancestor structural nodes (levels that never
    capture, e.g. the persona subgraph) are reconstructed by the consumer from
    the ``<name>:<uuid>`` segment prefixes.

    Args:
        checkpoint_ns: The raw ``checkpoint_ns`` string from LangGraph
            runtime config, or ``None``/empty for the root graph.

    Returns:
        A ``(id, parent_id)`` tuple. The root graph returns
        ``("__root__", None)``.
    """
    if not checkpoint_ns:
        return _ROOT, None
    segments = _strip_capture_segments(checkpoint_ns.split("|"))
    if not segments:
        return _ROOT, None
    node_id = "|".join(segments)
    if len(segments) == 1:
        return node_id, _ROOT
    return node_id, "|".join(segments[:-1])


def resolve_node_id(node_id: str, parent_id: str | None, kind: str) -> str:
    """Guarantee a unique id for a task-invoked subagent.

    Deepagent ``task`` subagents may not receive a ``checkpoint_ns`` distinct from
    their parent, which would collapse the derived id onto the parent's and let
    the reducer overwrite one node with the other. When ``kind == "task"`` and the
    derived id would duplicate the parent, mint a unique child id so each task
    invocation is its own node under that parent.
    """
    if kind == "task" and node_id == parent_id:
        return f"{parent_id}|task:{uuid.uuid4().hex[:8]}"
    return node_id


def _summarise(tool_runs: list[dict[str, Any]]) -> ToolSummary:
    """Aggregate a list of tool-run records into a ``ToolSummary``.

    Args:
        tool_runs: Tool-run records as produced by
            ``agent_capture.records.build_tool_records``.

    Returns:
        Counts of total/ok/failed/cached calls plus the distinct tool names
        (in first-seen order).
    """
    tools: list[str] = []
    ok = failed = cached = 0
    for r in tool_runs:
        name = r.get("tool")
        if name and name not in tools:
            tools.append(name)
        if r.get("status") == "error":
            failed += 1
        else:
            ok += 1
        if r.get("cache_hit"):
            cached += 1
    return {
        "count": len(tool_runs),
        "tools": tools,
        "ok": ok,
        "failed": failed,
        "cached": cached,
    }


def build_tree_node(
    *,
    node_id: str,
    parent_id: str | None,
    name: str,
    kind: Literal["subgraph", "task"],
    tool_runs: list[dict[str, Any]],
    output: Any,
) -> TreeNode:
    """Build a light topology node summarising what this agent did.

    Args:
        node_id: Stable id for this node (see ``node_ids_from_ns``).
        parent_id: Id of the parent node, or ``None`` for the root.
        name: Human-readable agent/node name.
        kind: Whether this node is a compiled ``subgraph`` or a ``task``-tool
            invocation.
        tool_runs: This agent's tool-run records, used to build the
            ``tool_summary``.
        output: The agent's final output (structured response or free-form
            content); truncated into ``output_preview``.

    Returns:
        A fully populated ``TreeNode``.
    """
    summary = _summarise(tool_runs or [])
    preview = None
    if output:
        preview = (str(output)[:280]) or None
    status: Literal["ok", "error"] = (
        "error" if summary["failed"] and not summary["ok"] else "ok"
    )
    return {
        "id": node_id,
        "parent_id": parent_id,
        "name": name,
        "kind": kind,
        "status": status,
        "tool_summary": summary,
        "output_preview": preview,
        "has_detail": True,
    }


def merge_subagent_tree(
    left: dict[str, TreeNode] | None, right: dict[str, TreeNode] | None
) -> dict[str, TreeNode]:
    """Reduce two partial trees into one, keyed by node id.

    Suitable as the reducer for an ``Annotated[dict[str, TreeNode], ...]``
    state channel: nested/parallel captures each contribute their own nodes,
    and this merge accumulates them by id (later writes win on key collision).

    Args:
        left: The existing accumulated tree, or ``None``.
        right: The incoming partial tree to merge in, or ``None``.

    Returns:
        The merged tree.
    """
    return {**(left or {}), **(right or {})}
