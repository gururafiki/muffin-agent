"""Subagent refinement middleware — generic conversational protocol.

When added to a subagent it forces a :class:`CollectionFindings`
structured response, caches it under
``/scratch/subagent_runs/<call_id>.json``, and on the next call (when
the parent passes ``prior_call_id=<id>`` in the description) re-injects
the prior findings into the subagent's system prompt so it can fill
gaps without restarting.

Two roles, two classes — pick one per agent:

* :class:`SubagentRefinementMiddleware` — child role; full hook surface.
* :class:`SubagentRefinementParentMiddleware` — parent role; only
  amends the system prompt with the orchestrator's gap-handling rules.

Internal layout:

* ``schema.py`` — ``CollectionFindings``, ``Gap``, ``GapReason``,
  ``ToolCallSummary``.
* ``storage.py`` — read/write JSON findings via the agent's backend +
  ``latest_human_text`` helper.
* ``prompts.py`` — thin facade over :func:`muffin_agent.prompts.render_template`
  that loads the role-specific instructions and renders the per-call
  prior-findings block.  Templates live under
  ``muffin_agent/prompts/middlewares/subagent_refinement/``.
* ``middleware.py`` — wires the three above into LangChain's
  ``before_agent`` / ``awrap_model_call`` / ``after_agent`` hooks.
"""

from .middleware import (
    SubagentRefinementMiddleware,
    SubagentRefinementParentMiddleware,
    SubagentRefinementState,
)
from .prompts import child_instructions, parent_instructions
from .schema import (
    CollectionFindings,
    Gap,
    GapReason,
    ToolCallSummary,
)

__all__ = [
    "CollectionFindings",
    "Gap",
    "GapReason",
    "SubagentRefinementMiddleware",
    "SubagentRefinementParentMiddleware",
    "SubagentRefinementState",
    "ToolCallSummary",
    "child_instructions",
    "parent_instructions",
]
