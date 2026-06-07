"""Specialist signal agents — deterministic siblings of personas.

Specialists emit the same :class:`AnalystSignal` contract as personas but
skip the LLM call. Their scoring is fully mechanical (technical
indicators, sentiment aggregation), so they're cheap, fast, and
deterministic.

Six specialists ship today. Two are fully deterministic (sync, no-arg
builders); four mirror the metric-heavy upstream specialists and use a
persona-style ReAct ``collect_data`` node purely for reliable extraction of
OpenBB ratio fields, followed by deterministic scoring (async builders taking
``config``, like personas):

* ``technicals`` — 5-strategy ensemble over the 1-year OHLCV series (sync).
* ``sentiment`` — 30/70 weighted insider + news sentiment aggregation (sync).
* ``fundamentals`` — 4-dimension multi-metric scoring (ReAct collect → compute).
* ``growth`` — 5 weighted sub-scores (growth/valuation/margins/insider/health).
* ``valuation`` — 4-method weighted intrinsic-value gap (DCF / owner earnings /
  EV-EBITDA / residual income).
* ``news_sentiment`` — LLM headline classification + deterministic aggregation.

The specialists are imported and wired directly by the council
(``council_graph.py``) and CLI; there is no central registry.
"""

from __future__ import annotations

from .fundamentals_analysis import (
    FundamentalsEvidence,
    FundamentalsSignal,
    build_fundamentals_analysis_agent,
)
from .growth_analysis import (
    GrowthEvidence,
    GrowthSignal,
    build_growth_analysis_agent,
)
from .news_sentiment_analysis import (
    NewsSentimentEvidence,
    NewsSentimentSignal,
    build_news_sentiment_analysis_agent,
)
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
from .valuation_analysis import (
    ValuationEvidence,
    ValuationSignal,
    build_valuation_analysis_agent,
)

__all__ = [
    "FundamentalsEvidence",
    "FundamentalsSignal",
    "GrowthEvidence",
    "GrowthSignal",
    "NewsSentimentEvidence",
    "NewsSentimentSignal",
    "SentimentEvidence",
    "SentimentSignal",
    "TechnicalEvidence",
    "TechnicalSignal",
    "ValuationEvidence",
    "ValuationSignal",
    "build_fundamentals_analysis_agent",
    "build_growth_analysis_agent",
    "build_news_sentiment_analysis_agent",
    "build_sentiment_analysis_agent",
    "build_technical_analysis_agent",
    "build_valuation_analysis_agent",
]
