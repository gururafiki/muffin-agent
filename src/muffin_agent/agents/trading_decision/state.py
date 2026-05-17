"""State schemas for the trading_decision module.

``TradingDecisionState`` is the top-level state passed through the full
``build_trading_decision_graph`` pipeline.

Pipeline-stage outputs land in dedicated state keys:

* PR 1: ``investment_debate`` (sub-state) + ``investment_judge``.
* PR 2: ``trader``.
* PR 3 (this release): ``risk_debate`` (sub-state) + ``portfolio_decision``.
* PR 4 (planned): ``past_reflections`` (string block injected from memory).

Sub-states (``InvestmentDebateState`` and ``RiskDebateState``) mirror
TradingAgents' debate-state pattern: append-only history strings plus a
``current_response`` / ``latest_speaker`` tagged with the speaker name
so the conditional router can alternate without parsing message bodies.
"""

from __future__ import annotations

from typing import Literal

from typing_extensions import TypedDict

from .schemas import AnalysisContext


class InvestmentDebateState(TypedDict, total=False):
    """Sub-state for the Bull vs Bear investment debate.

    Mirrors TradingAgents' ``InvestDebateState`` (TradingAgents/.../agents/
    utils/agent_states.py). Histories accumulate across turns; ``count`` is
    the exit signal for the alternation router.
    """

    history: str
    """Full interleaved transcript of Bull and Bear turns."""

    bull_history: str
    """Concatenation of every Bull Researcher turn."""

    bear_history: str
    """Concatenation of every Bear Researcher turn."""

    current_response: str
    """Latest argument. Prepended with ``"Bull Researcher: "`` or
    ``"Bear Researcher: "`` so the router can dispatch the opposite
    speaker without parsing the body."""

    judge_decision: str
    """Rendered judge synthesis (debate-state copy of
    ``state["investment_judge"]`` for transcript completeness)."""

    count: int
    """Number of debate turns taken. Exit when
    ``count >= 2 * max_investment_debate_rounds``."""


class RiskDebateState(TypedDict, total=False):
    """Sub-state for the 3-way Aggressive / Conservative / Neutral risk debate.

    Mirrors TradingAgents' ``RiskDebateState`` (TradingAgents/.../agents/
    utils/agent_states.py). Three round-robin participants each get their
    own history string plus a snapshot of their latest argument; the router
    uses the explicit ``latest_speaker`` field rather than parsing
    ``current_response``, which would require disambiguating three tag
    prefixes.
    """

    history: str
    """Full interleaved transcript of all three speakers' turns."""

    aggressive_history: str
    """Concatenation of every Aggressive Analyst turn."""

    conservative_history: str
    """Concatenation of every Conservative Analyst turn."""

    neutral_history: str
    """Concatenation of every Neutral Analyst turn."""

    current_aggressive_response: str
    """Latest Aggressive turn (speaker-tagged). Empty until the first turn."""

    current_conservative_response: str
    """Latest Conservative turn (speaker-tagged). Empty until the first turn."""

    current_neutral_response: str
    """Latest Neutral turn (speaker-tagged). Empty until the first turn."""

    latest_speaker: Literal[
        "Aggressive", "Conservative", "Neutral", "Portfolio Manager"
    ]
    """Who just spoke. Used by the round-robin router to pick the next
    debater. Set to ``"Portfolio Manager"`` once the synthesis judge runs."""

    judge_decision: str
    """Rendered Portfolio Manager decision (debate-state copy of
    ``state["portfolio_decision"]`` for transcript completeness)."""

    count: int
    """Number of debate turns taken. Exit when
    ``count >= 3 * max_risk_debate_rounds``."""


class TradingDecisionState(TypedDict, total=False):
    """Top-level state for ``build_trading_decision_graph``.

    Output keys accumulate across stages:

    * PR 1: ``investment_debate`` + ``investment_judge``.
    * PR 2: ``trader``.
    * PR 3 (this release): ``risk_debate`` + ``portfolio_decision``.
    * PR 4 (planned): ``past_reflections``.
    """

    # ── Input ───────────────────────────────────────────────────────────────
    analysis_context: AnalysisContext
    """Required input. Carries ticker, query, and any upstream analysis."""

    # ── Investment debate (PR 1) ────────────────────────────────────────────
    investment_debate: InvestmentDebateState
    investment_judge: dict
    """``InvestmentJudgeOutput.model_dump()`` or an error fallback dict."""

    # ── Trader (PR 2) ───────────────────────────────────────────────────────
    trader: dict
    """``TraderOutput.model_dump()`` or an error fallback dict."""

    # ── Risk debate + Portfolio Manager (PR 3) ──────────────────────────────
    risk_debate: RiskDebateState
    portfolio_decision: dict
    """``PortfolioDecisionOutput.model_dump()`` or an error fallback dict.
    This is the canonical final artifact of the trading-decision pipeline."""

    # ── Reflection memory (PR 4) ─────────────────────────────────────────────
    past_reflections: str
    """Pre-formatted Markdown block of past same-ticker and cross-ticker
    reflections, populated by ``reflector_resolve_node`` and injected into
    the Portfolio Manager prompt. Empty string when no resolved reflections
    are available (cold start, reflection disabled, store unavailable)."""

    decision_date: str
    """The ``YYYY-MM-DD`` date this decision is being made on. Defaults to
    today (UTC) when ``reflector_resolve_node`` runs. Carried through state
    so ``decision_writeback_node`` can use the same key as the resolver."""

    resolved_decisions: list[dict]
    """List of ``DecisionRecord.model_dump()`` payloads that
    ``reflector_resolve_node`` resolved this run (for observability /
    logging — not a load-bearing field)."""
