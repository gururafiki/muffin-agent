"""Per-run configuration for the trading_decision pipeline.

Mirrors muffin's other ``BaseConfiguration`` subclasses (``MemoryConfiguration``,
``McpConfiguration``, ``ResearchConfiguration``). All fields are populated from
``RunnableConfig["configurable"]`` with env-var fallback (UPPERCASE name).

Used by:

* Graph-level router functions to decide debate-round budgets.
* ``reflector_resolve_node`` / ``decision_writeback_node`` for reflection
  toggles and outcome-fetch parameters.
"""

from __future__ import annotations

from pydantic import Field

from ...utils.base_config import BaseConfiguration


class TradingDecisionConfiguration(BaseConfiguration):
    """Per-run knobs for ``build_trading_decision_graph`` and its variants."""

    max_investment_debate_rounds: int = Field(default=2, ge=1)
    """Bull↔Bear cycles. Default 2 rounds = 4 turns (allows one rebuttal pair)."""

    max_risk_debate_rounds: int = Field(default=1, ge=1)
    """Aggressive→Conservative→Neutral cycles. Default 1 round = 3 turns."""

    reflection_enabled: bool = True
    """Master switch for the reflection-memory bookends. When ``False`` the
    resolver and writeback nodes degrade to no-ops."""

    reflection_holding_days: int = Field(default=5, ge=1)
    """Trading-day window over which realised returns are computed."""

    reflection_benchmark: str = "SPY"
    """Ticker symbol used for the alpha calculation."""

    reflection_max_same_ticker: int = Field(default=5, ge=0)
    """Number of most-recent same-ticker resolved reflections injected into the
    Portfolio Manager prompt."""

    reflection_max_cross_ticker: int = Field(default=3, ge=0)
    """Number of most-recent cross-ticker resolved reflections injected into the
    Portfolio Manager prompt."""

    decision_date: str | None = None
    """``YYYY-MM-DD`` override for the date this decision is being made on.
    Used for deterministic testing; defaults to today UTC in the resolver."""
