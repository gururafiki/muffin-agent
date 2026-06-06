"""Specialist scaffolding — :class:`SpecialistSpec` + :data:`SPECIALIST_REGISTRY`.

Specialists are the deterministic siblings of personas: they emit the same
``AnalystSignal`` contract but skip the LLM call.  Their scoring is fully
mechanical (technical indicators, sentiment aggregation), so they're
cheap, fast, and produce identical results given the same inputs.

The registry mirrors :data:`muffin_agent.agents.personas.PERSONA_REGISTRY`
so the council graph can optionally fan-out to specialists alongside
personas (via ``--include-specialists`` on the CLI, future).
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from langchain_core.runnables import RunnableConfig

from ..personas._base import PersonaInputState, PersonaOutputState
from ..personas.schemas import AnalystSignal

SpecialistNode = Callable[
    [PersonaInputState, RunnableConfig], Awaitable[PersonaOutputState]
]
"""Type alias — specialist nodes share the persona input/output state
contract so they can be wired into the council graph alongside personas
via the same ``persona_signals`` reducer."""


@dataclass(frozen=True)
class SpecialistSpec:
    """Registry entry for one specialist signal agent.

    Specialists self-register at module-import time via
    :func:`register_specialist`.  Order is preserved by
    :data:`SPECIALIST_REGISTRY` (dict insertion order).
    """

    slug: str
    """Stable identifier (e.g. ``"technicals"``, ``"sentiment"``)."""

    display_name: str
    """Human-readable name."""

    investing_style: str
    """One-sentence summary of what the specialist measures."""

    node: SpecialistNode
    """The async node function — same signature as a persona node."""

    signal_schema: type[AnalystSignal]
    """Narrowed Pydantic signal schema for this specialist."""


SPECIALIST_REGISTRY: dict[str, SpecialistSpec] = {}
"""All registered specialists keyed by slug, in insertion order.

The council graph optionally fans out to these alongside personas when
``--include-specialists`` is set.
"""


def register_specialist(spec: SpecialistSpec) -> SpecialistSpec:
    """Add *spec* to :data:`SPECIALIST_REGISTRY` keyed by ``spec.slug``.

    Returns the spec unchanged so it can be used at module level.  Raises
    ``ValueError`` on duplicate slug.
    """
    if spec.slug in SPECIALIST_REGISTRY:
        raise ValueError(f"Specialist slug {spec.slug!r} already registered")
    SPECIALIST_REGISTRY[spec.slug] = spec
    return spec
