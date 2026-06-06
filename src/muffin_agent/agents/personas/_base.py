"""Persona scaffolding — :class:`PersonaSpec` + :data:`PERSONA_REGISTRY`.

Each persona file under ``agents/personas/<slug>.py`` exports its own
``PERSONA_SPEC`` constant.  They self-register into :data:`PERSONA_REGISTRY`
on import via the helper :func:`register_persona`.  The council graph
(Phase 2.4) iterates ``PERSONA_REGISTRY`` to fan out ``Send``s to all
personas; the CLI uses it to look up a persona by slug.

A persona is a single async node function with the signature::

    async def <slug>_node(
        state: PersonaInputState, config: RunnableConfig
    ) -> dict[str, list[dict]]:
        # 1. Compute facts from state["data_bundle"] via tools/scoring_helpers
        # 2. Single LLM call with structured output (<Persona>Signal)
        # 3. Return {"persona_signals": [signal.model_dump()]}

This shape matches muffin's existing ``portfolio_manager_node`` /
``investment_judge_node`` in ``trading_decision`` — a single LLM call
against a Pydantic schema, no ReAct loop, no subagents.
"""

from __future__ import annotations

import operator
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Annotated, Any

from langchain_core.runnables import RunnableConfig
from typing_extensions import TypedDict

from .schemas import AnalystSignal

# ── Persona node signature ────────────────────────────────────────────────────


class PersonaInputState(TypedDict, total=False):
    """State keys read by a persona node.

    Every persona reads ``data_bundle`` (produced by
    ``persona_data_collection_node``) plus ``ticker`` and an optional
    ``query`` (the investment mandate context).  Specific personas may
    additionally read state fields not declared here — they do so
    defensively with ``.get()``.
    """

    ticker: str
    query: str
    data_bundle: dict[str, Any]


class PersonaOutputState(TypedDict, total=False):
    """State keys written by a persona node.

    All personas write to the same ``persona_signals`` reducer so the
    council graph collects them via parallel ``Send`` fan-out.
    """

    persona_signals: Annotated[list[dict[str, Any]], operator.add]


PersonaNode = Callable[
    [PersonaInputState, RunnableConfig], Awaitable[PersonaOutputState]
]
"""Type alias for a persona node function."""


# ── PersonaSpec + registry ────────────────────────────────────────────────────


@dataclass(frozen=True)
class PersonaSpec:
    """Registry entry for one persona.

    Personas register themselves at module-import time by calling
    :func:`register_persona` with their ``PersonaSpec``.  Order of
    registration is preserved by :data:`PERSONA_REGISTRY` (a dict, which
    is insertion-ordered as of Python 3.7).
    """

    slug: str
    """Stable kebab-case-converted snake_case identifier used by the
    council fan-out, the CLI (``muffin persona <slug>``), and the
    ``AnalystSignal.agent_id`` field.  Examples: ``"warren_buffett"``,
    ``"cathie_wood"``."""

    display_name: str
    """Human-readable persona name (e.g. ``"Warren Buffett"``).  Used in
    Rich tables and prompt headers."""

    investing_style: str
    """One-sentence summary of the persona's lens (e.g. "Quality
    compounders at fair prices; 3-stage owner-earnings DCF; ≥20% MOS").
    Used in CLI listings and as Jinja context in the council judge
    prompt."""

    node: PersonaNode
    """The async node function (see :data:`PersonaNode`)."""

    signal_schema: type[AnalystSignal]
    """The persona's narrowed Pydantic signal schema (e.g.
    ``WarrenBuffettSignal``).  Used for tests, docs, and the standalone
    CLI's output formatting."""


PERSONA_REGISTRY: dict[str, PersonaSpec] = {}
"""All registered personas keyed by slug.  Populated at module-import
time as each persona file calls :func:`register_persona`.  The council
graph iterates this in declaration order for fan-out."""


def register_persona(spec: PersonaSpec) -> PersonaSpec:
    """Add *spec* to :data:`PERSONA_REGISTRY` keyed by ``spec.slug``.

    Returns the spec unchanged so it can be used in ``PERSONA_SPEC =
    register_persona(PersonaSpec(...))`` at module level.  Raises
    ``ValueError`` on duplicate slug — prevents silent registry
    collisions if two personas accidentally share a slug.
    """
    if spec.slug in PERSONA_REGISTRY:
        raise ValueError(f"Persona slug {spec.slug!r} already registered")
    PERSONA_REGISTRY[spec.slug] = spec
    return spec
