"""State types for the multi_agent conference framework.

`ConferenceState` is the default state schema for `build_conference_graph`.
Callers passing a wider schema (e.g. a parent graph's state) must either
include the same field names or override the framework's defaults via the
``transcript_field`` / ``next_speaker_field`` / ``verdict_field`` parameters
on ``build_conference_graph``.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any

from typing_extensions import TypedDict


class Turn(TypedDict):
    """One participant turn in a conference transcript."""

    speaker: str
    content: str
    round: int


class ConferenceState(TypedDict, total=False):
    """Default state schema for ``build_conference_graph``."""

    transcript: Annotated[list[Turn], operator.add]
    next_speaker: str | None
    verdict: dict[str, Any] | None
