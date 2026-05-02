"""Prompt content owned by the subagent-refinement middleware.

Templates live under ``muffin_agent/prompts/middlewares/subagent_refinement/``
and are rendered through the project's standard
:func:`muffin_agent.prompts.render_template` loader, so this module is just
a thin facade that picks the right template per call.
"""

from __future__ import annotations

import json

from langchain_core.messages import SystemMessage

from ...prompts import render_template
from .schema import CollectionFindings

_TEMPLATE_DIR = "middlewares/subagent_refinement"
_TEMPLATE_CHILD = f"{_TEMPLATE_DIR}/child_instructions.jinja"
_TEMPLATE_PARENT = f"{_TEMPLATE_DIR}/parent_instructions.jinja"
_TEMPLATE_PRIOR = f"{_TEMPLATE_DIR}/prior_findings_block.jinja"


def child_instructions() -> str:
    """Render the child-role static rules."""
    return render_template(_TEMPLATE_CHILD)


def parent_instructions() -> str:
    """Render the parent-role static rules."""
    return render_template(_TEMPLATE_PARENT)


def render_prior_findings_block(findings: CollectionFindings) -> str:
    """Render the system-prompt addendum that exposes a prior call's data."""
    obtained_json = json.dumps(
        findings.obtained, indent=2, sort_keys=True, default=str
    )
    gap_lines = [
        f"- {gap.field} — reason: {gap.reason.value}"
        + (f" (advice: {gap.retry_advice})" if gap.retry_advice else "")
        for gap in findings.gaps
    ]
    return render_template(
        _TEMPLATE_PRIOR,
        call_id=findings.call_id,
        obtained_json=obtained_json,
        gap_lines=gap_lines,
    )


def append_block(existing: SystemMessage | None, block: str) -> SystemMessage:
    """Append *block* to *existing*, preserving prior content where possible."""
    existing_text = (
        existing.content
        if existing is not None and isinstance(existing.content, str)
        else ""
    )
    combined = f"{existing_text}\n\n{block}".strip() if existing_text else block
    return SystemMessage(content=combined)
