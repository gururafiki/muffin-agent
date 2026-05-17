"""Conditional routing helpers for the trading_decision graph.

Pure-state inspectors that decide the next node in a debate. Mirrors
TradingAgents' speaker-tag + count-based alternation pattern (see
TradingAgents/.../graph/conditional_logic.py). No LLM calls, no parsing
beyond a prefix check; cheap and deterministic.

Per-run round counts live on ``RunnableConfig.configurable``:

* ``max_investment_debate_rounds`` (default 2) — Bull→Bear→Bull→Bear cycle
* ``max_risk_debate_rounds`` (default 1) — Agg→Cons→Neut cycle

LangGraph conditional edges receive **only** the state, not the config.
The routers therefore pull the active ``RunnableConfig`` from
``langgraph.config.get_config()``. Tests that exercise the routing logic
in isolation can pass an explicit configurable via the private
``_route_*`` helpers below.
"""

from __future__ import annotations

from typing import Any

# Speaker-tag prefixes. Each debater's node wrapper prepends these to its
# response so the router can dispatch without parsing the body.
BULL_TAG = "Bull Researcher:"
BEAR_TAG = "Bear Researcher:"

# Risk-debate speaker tags. The 3-way debate uses the explicit
# ``latest_speaker`` field (an enum) for routing rather than tag-prefix
# matching on ``current_response`` — disambiguating three prefixes against
# free-form prose is fragile, the enum is cheap.
AGGRESSIVE_TAG = "Aggressive Analyst:"
CONSERVATIVE_TAG = "Conservative Analyst:"
NEUTRAL_TAG = "Neutral Analyst:"

# Default round counts. Overridable per-run via ``RunnableConfig.configurable``.
DEFAULT_MAX_INVESTMENT_DEBATE_ROUNDS = 2
"""Bull→Bear→Bull→Bear (4 turns total). Allows one round of rebuttal."""

DEFAULT_MAX_RISK_DEBATE_ROUNDS = 1
"""Aggressive→Conservative→Neutral (3 turns total). One pass per persona —
risk debate is about perspective coverage, not adversarial rebuttal."""


def _active_configurable() -> dict[str, Any]:
    """Pull the ``configurable`` dict from the active LangGraph runtime, or empty.

    Routing functions are invoked by LangGraph with state only; the active
    ``RunnableConfig`` lives in a context-local set by the runtime. When
    called outside any graph context (unit tests for the helper itself),
    this returns an empty dict and callers fall back to defaults.
    """
    try:
        from langgraph.config import get_config

        config = get_config()
    except Exception:
        return {}
    return dict(config.get("configurable") or {})


def _route_investment_debate(
    state: dict[str, Any],
    configurable: dict[str, Any],
) -> str:
    """Pure routing logic — separated for direct testing without a graph context."""
    max_rounds = int(
        configurable.get(
            "max_investment_debate_rounds",
            DEFAULT_MAX_INVESTMENT_DEBATE_ROUNDS,
        )
    )
    debate = state.get("investment_debate") or {}
    count = int(debate.get("count", 0))
    if count >= 2 * max_rounds:
        return "investment_judge"
    current = str(debate.get("current_response") or "")
    if current.startswith(BULL_TAG):
        return "bear_researcher"
    return "bull_researcher"


def should_continue_investment_debate(state: dict[str, Any]) -> str:
    """Route the next investment-debate node.

    Returns one of:

    * ``"bear_researcher"`` — Bull just spoke, give Bear the floor.
    * ``"bull_researcher"`` — Bear just spoke (or debate is starting),
      give Bull the floor.
    * ``"investment_judge"`` — round budget exhausted; hand off to the
      synthesis judge.
    """
    return _route_investment_debate(state, _active_configurable())


def _route_risk_debate(
    state: dict[str, Any],
    configurable: dict[str, Any],
) -> str:
    """Pure routing logic for the 3-way risk debate."""
    max_rounds = int(
        configurable.get(
            "max_risk_debate_rounds",
            DEFAULT_MAX_RISK_DEBATE_ROUNDS,
        )
    )
    debate = state.get("risk_debate") or {}
    count = int(debate.get("count", 0))
    if count >= 3 * max_rounds:
        return "portfolio_manager"
    latest = str(debate.get("latest_speaker") or "")
    if latest.startswith("Aggressive"):
        return "conservative_debator"
    if latest.startswith("Conservative"):
        return "neutral_debator"
    # Neutral, Portfolio Manager, or empty → Aggressive opens / re-opens the round.
    return "aggressive_debator"


def should_continue_risk_debate(state: dict[str, Any]) -> str:
    """Route the next risk-debate node.

    Round-robin: Aggressive → Conservative → Neutral → (repeat or exit).
    Routes to ``"portfolio_manager"`` once
    ``count >= 3 * max_risk_debate_rounds``.
    """
    return _route_risk_debate(state, _active_configurable())
