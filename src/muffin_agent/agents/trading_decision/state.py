"""Flat state schema for the trading_decision module.

The package is fully self-contained — it does NOT depend on
``agents/investment/`` outputs. Inputs are caller-supplied (``ticker``,
``decision_date``, optional ``query`` / ``narrative``); the four analyst
agents fill in the analysis (``market_report``, ``fundamentals_report``,
``news_report``, ``sentiment_report``) before the Bull/Bear/Judge/
Trader/PM downstream nodes run.

The Bull/Bear debate still uses per-speaker ``Annotated[list[str],
operator.add]`` fields (not migrated yet — see the multi_agent framework
roadmap). The risk debate (Aggressive/Conservative/Neutral) is wired
through ``muffin_agent.multi_agent.build_conference_graph`` which
accumulates speaker-tagged ``Turn`` dicts into
``risk_debate_transcript`` and uses ``next_speaker`` for internal
routing.

Structured outputs (``investment_judge``, ``trader``, ``portfolio_decision``)
live in their own top-level dict fields populated by the synthesis/judge
nodes via ``Pydantic.model_dump()``.

Reflection fields (``past_reflections``, ``resolved_decisions``) are
populated by the reflection bookends.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any

from typing_extensions import TypedDict

from ...multi_agent import Turn


class TradingDecisionState(TypedDict, total=False):
    """Top-level state for ``build_trading_decision_graph`` and its variants.

    All fields are optional (``total=False``); each node reads only the
    slice it needs. Per-role node files declare narrower ``<Role>InputState``
    and ``<Role>OutputState`` TypedDicts to document their precise contract.
    """

    # ── Inputs (from caller) ────────────────────────────────────────────────
    ticker: str
    """Stock ticker symbol (e.g. ``AAPL``). Required. Read by every
    analyst agent and forwarded into Bull/Bear/Judge/Trader/PM prompts."""

    decision_date: str
    """``YYYY-MM-DD`` for the decision. Set by ``reflector_resolve_node``
    if absent (defaults to today). Doubles as the storage key for the
    reflection bookend (``decision_writeback_node``)."""

    query: str
    """Optional user framing question / mandate (e.g. ``"long-term hold
    candidate"``)."""

    narrative: str
    """Optional caller-supplied prior context / research notes. Rendered
    into the downstream Bull/Bear/Judge/Trader/PM prompts alongside the
    analyst reports."""

    # ── Analyst outputs (set by the 4 analyst agents) ──────────────────────
    market_report: str
    """Markdown technical-analysis report from ``market_analyst``."""

    fundamentals_report: str
    """Markdown fundamentals report from ``fundamentals_analyst``."""

    news_report: str
    """Markdown news + macro report from ``news_analyst``."""

    sentiment_report: str
    """Markdown social-sentiment report from ``social_analyst``."""

    # ── Investment debate ──────────────────────────────────────────────────
    investment_bull_responses: Annotated[list[str], operator.add]
    """Each Bull turn appended via the reducer. Latest is ``...[-1]``."""

    investment_bear_responses: Annotated[list[str], operator.add]
    """Each Bear turn appended via the reducer."""

    investment_judge: dict[str, Any]
    """``InvestmentJudgeOutput.model_dump()`` — set by the judge node."""

    # ── Trader ─────────────────────────────────────────────────────────────
    trader: dict[str, Any]
    """``TraderOutput.model_dump()`` — set by the trader node."""

    # ── Risk debate (wired via multi_agent.build_conference_graph) ─────────
    risk_debate_transcript: Annotated[list[Turn], operator.add]
    """Speaker-tagged ``Turn`` dicts accumulated by the risk-debate
    conference subgraph. Each turn carries ``speaker``, ``content``,
    ``round``. The Portfolio Manager reads this as the synthesis input."""

    next_speaker: str | None
    """Conference-subgraph routing field — written by the ``dispatch``
    node, consumed by the conference's conditional edge. Set to ``None``
    when the terminator fires (signalling end-of-conference)."""

    # ── Portfolio decision (canonical artifact) ────────────────────────────
    portfolio_decision: dict[str, Any]
    """``PortfolioDecisionOutput.model_dump()`` — set by the PM node.
    **Canonical final artifact** for downstream consumers."""

    # ── Reflection memory ──────────────────────────────────────────────────
    past_reflections: str
    """Pre-rendered Markdown block of past same-ticker + cross-ticker
    reflections, injected into the Portfolio Manager prompt."""

    resolved_decisions: list[dict[str, Any]]
    """Observability: list of decisions the resolver resolved this run."""
