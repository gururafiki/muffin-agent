"""Specialist signal agents — deterministic siblings of personas.

Specialists emit the same :class:`AnalystSignal` contract as personas but
skip the LLM call. Their scoring is fully mechanical (technical
indicators, sentiment aggregation), so they're cheap, fast, and
deterministic.

Two specialists ship today, each a compiled deterministic subgraph:

* ``technicals`` — 5-strategy ensemble (trend / mean-reversion / momentum /
  vol-regime / stat-arb) over the 1-year OHLCV series.
* ``sentiment`` — 30/70 weighted insider + news sentiment aggregation.

v4 architecture: the specialists are imported and wired directly by the
council (``personas/council_graph.py``) and CLI; there is no central
registry or single-specialist wrapper graph any more.
"""

from __future__ import annotations

from .sentiment_analysis import (
    SentimentEvidence,
    SentimentSignal,
    build_sentiment_analysis_agent,
)
from .technical_analysis import (
    TechnicalEvidence,
    TechnicalSignal,
    build_technical_analysis_agent,
)

__all__ = [
    "SentimentEvidence",
    "SentimentSignal",
    "TechnicalEvidence",
    "TechnicalSignal",
    "build_sentiment_analysis_agent",
    "build_technical_analysis_agent",
]
