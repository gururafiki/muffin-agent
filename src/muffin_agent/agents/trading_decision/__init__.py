"""Trading decision pipeline — LangGraph-native composable building blocks.

Public surface:

* Graph builders:
  * :func:`build_investment_debate_graph` — Bull ↔ Bear → Investment Judge
  * :func:`build_investment_thesis_graph` — debate → Judge → Trader
  * :func:`build_trading_decision_graph` — full pipeline with reflection bookends

* Per-role nodes (each takes a typed input state, returns a typed output state):
  ``bull_researcher_node``, ``bear_researcher_node``, ``investment_judge_node``,
  ``trader_node``, ``aggressive_debator_node``, ``conservative_debator_node``,
  ``neutral_debator_node``, ``portfolio_manager_node``,
  ``reflector_resolve_node``, ``decision_writeback_node``.

* Per-role state schemas (TypedDicts): ``<Role>InputState`` + ``<Role>OutputState``
  for every node above. External callers can satisfy the input shape to reuse a
  node in another graph.

* Reflection helpers: :class:`ReflectionMemory`, :func:`reflect_on_decision`,
  :class:`OutcomesFetcher`, :func:`fetch_outcomes_openbb`,
  :func:`render_reflections_block`.

* Configuration: :class:`TradingDecisionConfiguration` (typed per-run knobs).

* Pydantic output schemas: :class:`AnalysisContext`, :class:`InvestmentJudgeOutput`,
  :class:`TraderOutput`, :class:`PortfolioDecisionOutput`, :class:`Outcome`,
  :class:`DecisionRecord`.

See ``docs/trading-decision.md`` for composition patterns and future
migration paths (subgraph with ``ToolNode`` vs ``MuffinAgentBuilder`` agent
as graph node).
"""

from .config import TradingDecisionConfiguration
from .graph import (
    build_investment_debate_graph,
    build_investment_thesis_graph,
    build_trading_decision_graph,
)
from .portfolio_manager import (
    PortfolioManagerInputState,
    PortfolioManagerOutputState,
    portfolio_manager_node,
)
from .reflection import (
    DecisionWritebackInputState,
    DecisionWritebackOutputState,
    OutcomesFetcher,
    ReflectionMemory,
    ReflectorResolveInputState,
    ReflectorResolveOutputState,
    decision_writeback_node,
    fetch_outcomes_openbb,
    reflect_on_decision,
    reflector_resolve_node,
    render_reflections_block,
)
from .researchers import (
    BearResearcherInputState,
    BearResearcherOutputState,
    BullResearcherInputState,
    BullResearcherOutputState,
    InvestmentJudgeInputState,
    InvestmentJudgeOutputState,
    bear_researcher_node,
    bull_researcher_node,
    investment_judge_node,
)
from .risk_debate import (
    AggressiveDebatorInputState,
    AggressiveDebatorOutputState,
    ConservativeDebatorInputState,
    ConservativeDebatorOutputState,
    NeutralDebatorInputState,
    NeutralDebatorOutputState,
    aggressive_debator_node,
    conservative_debator_node,
    neutral_debator_node,
)
from .schemas import (
    AnalysisContext,
    DecisionRecord,
    InvestmentJudgeOutput,
    InvestmentSignal,
    Outcome,
    PortfolioDecisionOutput,
    TraderAction,
    TraderOutput,
)
from .state import TradingDecisionState
from .trader import TraderInputState, TraderOutputState, trader_node

__all__ = [
    # Schemas
    "AnalysisContext",
    "DecisionRecord",
    "InvestmentJudgeOutput",
    "InvestmentSignal",
    "Outcome",
    "PortfolioDecisionOutput",
    "TraderAction",
    "TraderOutput",
    # Configuration
    "TradingDecisionConfiguration",
    # State
    "TradingDecisionState",
    # Graph builders
    "build_investment_debate_graph",
    "build_investment_thesis_graph",
    "build_trading_decision_graph",
    # Reflection helpers
    "OutcomesFetcher",
    "ReflectionMemory",
    "fetch_outcomes_openbb",
    "reflect_on_decision",
    "render_reflections_block",
    # Nodes
    "aggressive_debator_node",
    "bear_researcher_node",
    "bull_researcher_node",
    "conservative_debator_node",
    "decision_writeback_node",
    "investment_judge_node",
    "neutral_debator_node",
    "portfolio_manager_node",
    "reflector_resolve_node",
    "trader_node",
    # Per-role state TypedDicts
    "AggressiveDebatorInputState",
    "AggressiveDebatorOutputState",
    "BearResearcherInputState",
    "BearResearcherOutputState",
    "BullResearcherInputState",
    "BullResearcherOutputState",
    "ConservativeDebatorInputState",
    "ConservativeDebatorOutputState",
    "DecisionWritebackInputState",
    "DecisionWritebackOutputState",
    "InvestmentJudgeInputState",
    "InvestmentJudgeOutputState",
    "NeutralDebatorInputState",
    "NeutralDebatorOutputState",
    "PortfolioManagerInputState",
    "PortfolioManagerOutputState",
    "ReflectorResolveInputState",
    "ReflectorResolveOutputState",
    "TraderInputState",
    "TraderOutputState",
]
