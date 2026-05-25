"""Outcome-driven reflection loop for trading_decision.

Three pieces, layered:

    typed CRUD (per-user store)       : memory.py
    bookend graph nodes               : resolver.py, writeback.py
    outcome fetching                  : ../tools.py:fetch_decision_outcome
                                        (alongside get_indicators)
    LLM reflection helper             : inlined in resolver.py (single caller)
    user_id resolution                : utils.memory_config.MemoryConfiguration
                                        .resolve_user_id
    LLM resolution                    : model_config.ModelConfiguration
                                        .get_chat_model_for_role

Graph topology::

    START → reflector_resolve → [analysts fan-out] → ... → portfolio_manager
            ↓                                              ↓
            injects past_reflections                       writes
            into PM prompt                                 portfolio_decision
                                                           ↓
                                                           decision_writeback → END

Cross-run persistence is required because realised returns / alpha take
trading days to materialise (default holding window: 5 trading days,
``TradingDecisionConfiguration.reflection_holding_days``). The whole
value of the loop is cross-run calibration of the Portfolio Manager.

This package stays co-located with trading_decision because it has a
single consumer today. ``ReflectionMemory``'s CRUD pattern,
``OutcomesFetcher``'s Protocol seam (in ``../tools.py``), and the
bookend-node pair are all written generically and become extraction
candidates when a second pipeline (e.g. ``criteria_analysis``,
``investment_analysis``) needs the same write-pending →
resolve-with-outcomes → inject-past-lessons loop. The extraction path,
documented in ``docs/trading-decision.md``: move ``ReflectionMemory``
to ``utils/`` with the namespace leaf parameterised, then promote a
``DecisionReflectionMiddleware`` so any structured-output agent can
opt into self-learning via
``MuffinAgentBuilder.with_decision_reflection(...)``.
"""

from .memory import (
    ReflectionMemory,
    make_key,
    render_reflections_block,
    split_key,
    try_build_reflection_memory,
)
from .resolver import (
    ReflectorResolveInputState,
    ReflectorResolveOutputState,
    reflector_resolve_node,
)
from .writeback import (
    DecisionWritebackInputState,
    DecisionWritebackOutputState,
    decision_writeback_node,
)

__all__ = [
    "DecisionWritebackInputState",
    "DecisionWritebackOutputState",
    "ReflectionMemory",
    "ReflectorResolveInputState",
    "ReflectorResolveOutputState",
    "decision_writeback_node",
    "make_key",
    "reflector_resolve_node",
    "render_reflections_block",
    "split_key",
    "try_build_reflection_memory",
]
