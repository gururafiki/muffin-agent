"""Flat state schema for the trading_decision module.

Each debater's responses accumulate in a top-level ``Annotated[list[str],
operator.add]`` field. Nodes return ``{"<field>": [new_response]}`` and
LangGraph's reducer appends to the existing list. The "latest response" is
``state[<field>][-1]``; the round count is ``len(bulls) + len(bears)`` (etc.).

Structured outputs (``investment_judge``, ``trader``, ``portfolio_decision``)
live in their own top-level dict fields populated by the synthesis/judge
nodes via ``Pydantic.model_dump()``.

Reflection fields (``decision_date``, ``past_reflections``,
``resolved_decisions``) are populated by the reflection bookends.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any

from typing_extensions import TypedDict


class TradingDecisionState(TypedDict, total=False):
    """Top-level state for ``build_trading_decision_graph`` and its variants.

    All fields are optional (``total=False``); each node reads only the
    slice it needs. Per-role node files declare narrower ``<Role>InputState``
    and ``<Role>OutputState`` TypedDicts to document their precise contract.
    """

    # ── Input ───────────────────────────────────────────────────────────────
    analysis_context: dict[str, Any]
    """``AnalysisContext.model_dump()`` — the generic envelope for upstream
    analysis. Read by every LLM-call node."""

    # ── Investment debate (PR 1 / refactor) ─────────────────────────────────
    investment_bull_responses: Annotated[list[str], operator.add]
    """Each Bull turn appended via the reducer. Latest is ``...[-1]``."""

    investment_bear_responses: Annotated[list[str], operator.add]
    """Each Bear turn appended via the reducer."""

    investment_judge: dict[str, Any]
    """``InvestmentJudgeOutput.model_dump()`` — set by the judge node."""

    # ── Trader (PR 2 / refactor) ────────────────────────────────────────────
    trader: dict[str, Any]
    """``TraderOutput.model_dump()`` — set by the trader node."""

    # ── Risk debate (PR 3 / refactor) ───────────────────────────────────────
    risk_aggressive_responses: Annotated[list[str], operator.add]
    risk_conservative_responses: Annotated[list[str], operator.add]
    risk_neutral_responses: Annotated[list[str], operator.add]

    # ── Portfolio decision (canonical artifact) ─────────────────────────────
    portfolio_decision: dict[str, Any]
    """``PortfolioDecisionOutput.model_dump()`` — set by the PM node.
    **Canonical final artifact** for downstream consumers."""

    # ── Reflection memory (PR 4) ────────────────────────────────────────────
    decision_date: str
    """``YYYY-MM-DD`` resolved by ``reflector_resolve_node`` and reused by
    ``decision_writeback_node`` as the storage key."""

    past_reflections: str
    """Pre-rendered Markdown block of past same-ticker + cross-ticker
    reflections, injected into the Portfolio Manager prompt."""

    resolved_decisions: list[dict[str, Any]]
    """Observability: list of decisions the resolver resolved this run."""
