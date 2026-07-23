"""Unified agent capture from one message-walk.

Captures the transcript (``subagent_runs``) + tool records (``tool_runs``) +
the sub-agent execution tree (``subagent_tree``). Replaces the former
``subagent_transcript`` and ``tool_telemetry`` middlewares.
"""

from .detail_store import offload_subagent_detail
from .middleware import (
    AgentCaptureMiddleware,
    AgentCaptureParentMiddleware,
    AgentCaptureState,
    merge_subagent_runs,
)
from .records import (
    DEFAULT_EXCLUDE_TOOLS,
    build_tool_records,
    merge_tool_runs,
)
from .serialize import serialize_messages
from .tree import (
    TreeNode,
    build_tree_node,
    merge_subagent_tree,
    node_ids_from_ns,
)

__all__ = [
    "AgentCaptureMiddleware",
    "AgentCaptureParentMiddleware",
    "AgentCaptureState",
    "DEFAULT_EXCLUDE_TOOLS",
    "TreeNode",
    "build_tool_records",
    "build_tree_node",
    "merge_subagent_runs",
    "merge_subagent_tree",
    "merge_tool_runs",
    "node_ids_from_ns",
    "offload_subagent_detail",
    "serialize_messages",
]
