"""State schemas for the trading_decision module.

``TradingDecisionState`` is the top-level state passed through the full
``build_trading_decision_graph`` pipeline. PR 1 only populates the
``analysis_context`` input field and the ``investment_debate`` /
``investment_judge`` output fields; later PRs add ``trader``,
``risk_debate``, ``portfolio_decision``, and ``past_reflections``.

Sub-states (``InvestmentDebateState`` and ``RiskDebateState``) mirror
TradingAgents' debate-state pattern: append-only history strings plus a
``current_response`` tagged with the speaker name so the conditional
router can alternate without parsing message bodies.
"""

from __future__ import annotations

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


class TradingDecisionState(TypedDict, total=False):
    """Top-level state for ``build_trading_decision_graph``.

    PR 1 surface only includes ``analysis_context``, ``investment_debate``,
    and ``investment_judge``. Later PRs add ``trader``, ``risk_debate``,
    ``portfolio_decision``, and ``past_reflections`` fields.
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
