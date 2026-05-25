"""Trading decision pipeline — LangGraph-native composable building blocks.

The package is fully **self-contained**: it does NOT consume outputs from
muffin's other agents (``agents/investment/`` etc.). It fetches its own
data through OpenBB MCP (price / fundamentals / news / ownership) and
Firecrawl MCP (social / web) via four ReAct **analyst agents** that
produce free-text prose reports consumed downstream.

Public surface:

* Graph builders (async — accept ``RunnableConfig`` so the analyst
  ReAct agents can be built at graph-construction time):

  * :func:`build_investment_debate_graph` — analysts → Bull ↔ Bear →
    Investment Judge.
  * :func:`build_investment_thesis_graph` — analysts → debate → Judge
    → Trader.
  * :func:`build_trading_decision_graph` — full pipeline with reflection
    bookends and the 3-way risk debate + Portfolio Manager.

* Analyst factories (compiled ReAct agents added directly as
  parent-graph nodes): :func:`build_market_analyst_agent`,
  :func:`build_fundamentals_analyst_agent`,
  :func:`build_news_analyst_agent`, :func:`build_social_analyst_agent`.

* Per-role downstream nodes (each takes a typed input state, returns a
  typed output state): ``bull_researcher_node``,
  ``bear_researcher_node``, ``investment_judge_node``, ``trader_node``,
  ``portfolio_manager_node``, ``reflector_resolve_node``,
  ``decision_writeback_node``. The 3-way risk debate has migrated to
  the multi_agent conference framework — it no longer exposes per-role
  node functions; the conference subgraph is built inline in
  :mod:`graph`.

* Per-role state schemas (TypedDicts):
  ``<Role>InputState`` / ``<Role>OutputState`` for each downstream
  node, plus ``<Role>AnalystState`` (extending ``AgentState``) and
  ``<Role>AnalystOutput`` (Pydantic) for each analyst.

* In-process tool: :func:`get_indicators` — fills the
  technical-indicator gap left by OpenBB MCP. Used by the Market
  analyst.

* Reflection helpers: :class:`ReflectionMemory`,
  :func:`render_reflections_block`. Outcome fetching lives next to
  :func:`get_indicators` in ``tools.py`` as :class:`OutcomesFetcher`
  Protocol + default :func:`fetch_decision_outcome` implementation.

* Configuration: :class:`TradingDecisionConfiguration` (typed per-run
  knobs).

* Pydantic output schemas: :class:`InvestmentJudgeOutput`,
  :class:`TraderOutput`, :class:`PortfolioDecisionOutput`,
  :class:`Outcome`, :class:`DecisionRecord`.

See ``docs/trading-decision.md`` for the canonical "compiled agent as
direct parent-graph node" pattern this package uses and future
migration paths.
"""

from .analysts import (
    FundamentalsAnalystOutput,
    FundamentalsAnalystState,
    MarketAnalystOutput,
    MarketAnalystState,
    NewsAnalystOutput,
    NewsAnalystState,
    SocialAnalystOutput,
    SocialAnalystState,
    build_fundamentals_analyst_agent,
    build_market_analyst_agent,
    build_news_analyst_agent,
    build_social_analyst_agent,
)
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
    ReflectionMemory,
    ReflectorResolveInputState,
    ReflectorResolveOutputState,
    decision_writeback_node,
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
from .schemas import (
    DecisionRecord,
    InvestmentJudgeOutput,
    InvestmentSignal,
    Outcome,
    PortfolioDecisionOutput,
    TraderAction,
    TraderOutput,
)
from .state import TradingDecisionState
from .tools import OutcomesFetcher, fetch_decision_outcome, get_indicators
from .trader import TraderInputState, TraderOutputState, trader_node

__all__ = [
    # Schemas (synthesis / judge / trader / PM)
    "DecisionRecord",
    "InvestmentJudgeOutput",
    "InvestmentSignal",
    "Outcome",
    "PortfolioDecisionOutput",
    "TraderAction",
    "TraderOutput",
    # Analyst outputs (Pydantic)
    "FundamentalsAnalystOutput",
    "MarketAnalystOutput",
    "NewsAnalystOutput",
    "SocialAnalystOutput",
    # Analyst state schemas (extend AgentState)
    "FundamentalsAnalystState",
    "MarketAnalystState",
    "NewsAnalystState",
    "SocialAnalystState",
    # Configuration
    "TradingDecisionConfiguration",
    # State
    "TradingDecisionState",
    # Graph builders (async)
    "build_investment_debate_graph",
    "build_investment_thesis_graph",
    "build_trading_decision_graph",
    # Analyst factories (async)
    "build_fundamentals_analyst_agent",
    "build_market_analyst_agent",
    "build_news_analyst_agent",
    "build_social_analyst_agent",
    # Local tools / outcome fetcher
    "OutcomesFetcher",
    "fetch_decision_outcome",
    "get_indicators",
    # Reflection helpers
    "ReflectionMemory",
    "render_reflections_block",
    # Downstream nodes
    "bear_researcher_node",
    "bull_researcher_node",
    "decision_writeback_node",
    "investment_judge_node",
    "portfolio_manager_node",
    "reflector_resolve_node",
    "trader_node",
    # Per-role state TypedDicts
    "BearResearcherInputState",
    "BearResearcherOutputState",
    "BullResearcherInputState",
    "BullResearcherOutputState",
    "DecisionWritebackInputState",
    "DecisionWritebackOutputState",
    "InvestmentJudgeInputState",
    "InvestmentJudgeOutputState",
    "PortfolioManagerInputState",
    "PortfolioManagerOutputState",
    "ReflectorResolveInputState",
    "ReflectorResolveOutputState",
    "TraderInputState",
    "TraderOutputState",
]
