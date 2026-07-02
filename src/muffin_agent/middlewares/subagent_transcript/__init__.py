"""Subagent-transcript middleware.

Captures a subagent's own message transcript (its tool calls + reasoning) into
the ``subagent_runs`` state key so it merges up into the parent deep-agent's
thread state (deepagents' ``task`` tool forwards every non-excluded state key
from the subagent result). This makes each subagent's *internal* steps visible
when the parent run is reopened from history — deepagents subagents otherwise
run ephemerally and only their final report survives.

Two roles (mirrors ``subagent_refinement``):

* **Child** (:class:`SubagentTranscriptMiddleware`, registered on ReAct
  subagents) — on ``after_agent`` writes a trimmed transcript into
  ``subagent_runs[<run_id>]``.
* **Parent** (:class:`SubagentTranscriptParentMiddleware`, registered on deep
  orchestrators) — only declares the ``subagent_runs`` state key (with an
  accumulating reducer) so the merged-up child records land in thread state.
"""

from .middleware import (
    SubagentTranscriptMiddleware,
    SubagentTranscriptParentMiddleware,
    SubagentTranscriptState,
    merge_subagent_runs,
)

__all__ = [
    "SubagentTranscriptMiddleware",
    "SubagentTranscriptParentMiddleware",
    "SubagentTranscriptState",
    "merge_subagent_runs",
]
