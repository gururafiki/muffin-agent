"""Stage 4a: deterministic merge of skill-derived and web-derived criteria.

Pure Python — no LLM call.  Concatenates the two upstream criterion
lists, drops exact duplicates by canonical name (skill version wins),
re-normalises weights to 1.0, and tags each surviving criterion with
its source.  Predictable, cheap, debuggable.
"""

import re
from typing import Any

from langchain_core.runnables import RunnableConfig

from .state import CriteriaAnalysisState

# ── Canonicalisation ──────────────────────────────────────────────────────────


_PUNCT = re.compile(r"[^a-z0-9]+")


def _canonical_name(name: str) -> str:
    """Lowercase + collapse non-alphanumerics → underscore.

    ``"P/E Ratio"`` and ``"P / E ratio"`` both canonicalise to ``"p_e_ratio"``.
    """
    return _PUNCT.sub("_", name.lower()).strip("_")


# ── Merge ────────────────────────────────────────────────────────────────────


def merge_criteria_lists(
    skill_criteria: list[dict[str, Any]],
    web_criteria: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Deduplicate and source-tag two criterion lists.

    Args:
        skill_criteria: ``ValuationCriterion`` dicts from
            ``criteria_definition.criteria``.  Win on duplicate-name ties
            because their weights are tuned to the sector.
        web_criteria: ``ValuationCriterion`` dicts from
            ``valuation_methodology.additional_criteria``.

    Returns:
        Single list of criterion dicts.  Each entry is the original dict
        with an added ``source`` key (``"skill"`` or ``"web"``) and a
        re-normalised ``weight`` such that all weights sum to 1.0
        (within float tolerance).
    """
    seen: dict[str, dict[str, Any]] = {}

    # Skill criteria first so they win ties.
    for entry in skill_criteria or []:
        canonical = _canonical_name(entry.get("name", ""))
        if canonical and canonical not in seen:
            seen[canonical] = {**entry, "source": "skill"}

    for entry in web_criteria or []:
        canonical = _canonical_name(entry.get("name", ""))
        if canonical and canonical not in seen:
            seen[canonical] = {**entry, "source": "web"}

    merged = list(seen.values())

    total = sum(float(c.get("weight", 0.0)) for c in merged)
    if total > 0:
        for entry in merged:
            entry["weight"] = float(entry.get("weight", 0.0)) / total
    elif merged:
        # All upstream weights were zero or missing — give equal weight.
        equal = 1.0 / len(merged)
        for entry in merged:
            entry["weight"] = equal

    return merged


# ── Node ──────────────────────────────────────────────────────────────────────


async def merge_criteria_node(
    state: CriteriaAnalysisState,
    config: RunnableConfig,
) -> dict[str, Any]:
    """Stage 4a: produce the merged criteria list from Stages 2 and 3."""
    skill_criteria = (state.get("criteria_definition") or {}).get("criteria") or []
    web_criteria = (state.get("valuation_methodology") or {}).get(
        "additional_criteria"
    ) or []
    merged = merge_criteria_lists(skill_criteria, web_criteria)
    return {"merged_criteria": merged}
