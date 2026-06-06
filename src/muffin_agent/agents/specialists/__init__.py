"""Specialist signal agents — deterministic siblings of personas.

Specialists emit the same :class:`AnalystSignal` contract as personas but
skip the LLM call.  Their scoring is fully mechanical (technical
indicators, sentiment aggregation), so they're cheap, fast, and
deterministic.

Two specialists ship today:

* ``technicals`` — 5-strategy ensemble (trend / mean-reversion / momentum /
  vol-regime / stat-arb) over the 1-year OHLCV series.
* ``sentiment`` — 30/70 weighted insider + news sentiment aggregation.

Each self-registers into :data:`SPECIALIST_REGISTRY` on package import via
:func:`register_specialist`.
"""

from __future__ import annotations

# Side-effect imports populate SPECIALIST_REGISTRY.
from . import sentiment_analysis as _sentiment_analysis  # noqa: F401
from . import technical_analysis as _technical_analysis  # noqa: F401
from ._base import (
    SPECIALIST_REGISTRY,
    SpecialistNode,
    SpecialistSpec,
    register_specialist,
)
from .sentiment_analysis import (
    SentimentEvidence,
    SentimentSignal,
    sentiment_analysis_node,
)
from .single_specialist_graph import (
    SingleSpecialistState,
    build_single_specialist_graph,
)
from .technical_analysis import (
    TechnicalEvidence,
    TechnicalSignal,
    technical_analysis_node,
)

__all__ = [
    "SPECIALIST_REGISTRY",
    "SentimentEvidence",
    "SentimentSignal",
    "SingleSpecialistState",
    "SpecialistNode",
    "SpecialistSpec",
    "TechnicalEvidence",
    "TechnicalSignal",
    "build_single_specialist_graph",
    "register_specialist",
    "sentiment_analysis_node",
    "technical_analysis_node",
]
