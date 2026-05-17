"""Trading decision pipeline — composable building blocks ported from TradingAgents.

Public surface (PR 1 + PR 2 + PR 3 + PR 4):

* :func:`build_investment_debate_graph` — Bull ↔ Bear debate → Investment Judge
* :func:`build_investment_thesis_graph` — debate → Judge → Trader
* :func:`build_trading_decision_graph` — full pipeline with reflection memory
  bookends (reflector_resolve → debate → Judge → Trader → risk debate →
  Portfolio Manager → decision_writeback)
* :class:`AnalysisContext` — generic envelope for upstream analysis context
* Structured outputs: :class:`InvestmentJudgeOutput`, :class:`TraderOutput`,
  :class:`PortfolioDecisionOutput`, :class:`Outcome`, :class:`DecisionRecord`
* State schemas: :class:`TradingDecisionState`, :class:`InvestmentDebateState`,
  :class:`RiskDebateState`
* :class:`ReflectionMemory` — per-user persistent decision log
* :class:`OutcomesFetcher` — protocol for realised-return data sources
* Standalone agent factories: ``create_{bull_researcher,bear_researcher,
  investment_judge,trader,aggressive_debator,conservative_debator,
  neutral_debator,portfolio_manager,reflector}_agent``

See ``CLAUDE.md`` and the plan at
``~/.claude/plans/explore-in-depth-agents-from-curried-muffin.md`` for the
broader architecture.
"""

from .graph import (
    build_investment_debate_graph,
    build_investment_thesis_graph,
    build_trading_decision_graph,
)
from .portfolio_manager import create_portfolio_manager_agent
from .reflection import (
    OutcomesFetcher,
    ReflectionMemory,
    create_reflector_agent,
    fetch_outcomes_openbb,
    generate_reflection,
    render_reflections_block,
)
from .researchers import (
    create_bear_researcher_agent,
    create_bull_researcher_agent,
    create_investment_judge_agent,
)
from .risk_debate import (
    create_aggressive_debator_agent,
    create_conservative_debator_agent,
    create_neutral_debator_agent,
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
from .state import InvestmentDebateState, RiskDebateState, TradingDecisionState
from .trader import create_trader_agent

__all__ = [
    "AnalysisContext",
    "DecisionRecord",
    "InvestmentDebateState",
    "InvestmentJudgeOutput",
    "InvestmentSignal",
    "Outcome",
    "OutcomesFetcher",
    "PortfolioDecisionOutput",
    "ReflectionMemory",
    "RiskDebateState",
    "TraderAction",
    "TraderOutput",
    "TradingDecisionState",
    "build_investment_debate_graph",
    "build_investment_thesis_graph",
    "build_trading_decision_graph",
    "create_aggressive_debator_agent",
    "create_bear_researcher_agent",
    "create_bull_researcher_agent",
    "create_conservative_debator_agent",
    "create_investment_judge_agent",
    "create_neutral_debator_agent",
    "create_portfolio_manager_agent",
    "create_reflector_agent",
    "create_trader_agent",
    "fetch_outcomes_openbb",
    "generate_reflection",
    "render_reflections_block",
]
