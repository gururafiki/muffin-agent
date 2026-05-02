"""Generic structured response for any data-collection subagent.

This is the contract every subagent emits via
:class:`SubagentRefinementMiddleware`'s injected ``response_format``. The
parent reads ``gaps`` (and optionally ``confidence``) to decide whether
to continue, retry with a refined ask, or accept partial data.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field


class GapReason(StrEnum):
    """Why a requested field could not be returned."""

    NO_DATA = "no_data"  # provider has no data for this entity
    PROVIDER_UNAVAILABLE = "provider_unavailable"  # missing key, quota exhausted
    NOT_ATTEMPTED = "not_attempted"  # ran out of budget / scope
    AMBIGUOUS_REQUEST = "ambiguous_request"  # could not decide what was asked
    UPSTREAM_ERROR = "upstream_error"  # transient; may recover on retry


class Gap(BaseModel):
    """One missing field plus the reason it couldn't be filled."""

    field: str = Field(description="Name of the requested field that is missing.")
    reason: GapReason = Field(description="Why the field is missing.")
    detail: str | None = Field(
        default=None,
        description="Short human note about the failure (e.g. 'FMP 402 quota').",
    )
    retry_advice: Literal["retry", "switch_tool", "give_up"] | None = Field(
        default=None,
        description=(
            "Hint for the parent: 'retry' if the subagent thinks a re-call "
            "may succeed, 'switch_tool' if a different tool would help, "
            "'give_up' if no further attempt is worthwhile."
        ),
    )


class ToolCallSummary(BaseModel):
    """Short trace of a tool call for parent observability."""

    tool: str
    status: Literal["success", "error"]
    error_class: str | None = Field(
        default=None,
        description="Populated when status=='error'; e.g. 'HTTP 422'.",
    )


class CollectionFindings(BaseModel):
    """Generic structured response any subagent emits.

    The parent agent uses ``call_id`` to refer back to this run when it
    needs the subagent to fill remaining gaps without re-running the
    full task. ``obtained`` is intentionally a free-form ``dict`` so
    each subagent decides which field names it returns; ``requested``
    + ``gaps`` together describe what's missing.
    """

    call_id: str = Field(
        description=(
            "Unique identifier for this subagent run. The parent passes "
            "it back as `prior_call_id=...` to refine without restarting."
        ),
    )
    requested: list[str] = Field(
        default_factory=list,
        description="Field names the parent asked for in this call.",
    )
    obtained: dict[str, Any] = Field(
        default_factory=dict,
        description="Field name → value for fields that were collected.",
    )
    gaps: list[Gap] = Field(
        default_factory=list,
        description="Fields that could not be filled, with reason & advice.",
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Subagent's overall confidence in `obtained`.",
    )
    tools_used: list[ToolCallSummary] = Field(
        default_factory=list,
        description="Tool calls the subagent made (for parent observability).",
    )
    notes: str | None = Field(
        default=None,
        description="Free-text notes from the subagent (caveats, sources).",
    )
