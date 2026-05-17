"""Trading decision pipeline — composable building blocks ported from TradingAgents.

Public surface (PR 1 + PR 2):

* :func:`build_investment_debate_graph` — Bull ↔ Bear debate → Investment Judge
* :func:`build_investment_thesis_graph` — debate → Judge → Trader
* :class:`AnalysisContext` — generic envelope for upstream analysis context
* :class:`InvestmentJudgeOutput` / :class:`TraderOutput` — structured outputs
* :class:`TradingDecisionState` / :class:`InvestmentDebateState` — state schemas
* Standalone agent factories: ``create_{bull_researcher,bear_researcher,
  investment_judge,trader}_agent``

Later PRs add the Risk Debate, Portfolio Manager, and reflection memory
layers without changing the existing surface.

See ``CLAUDE.md`` and the plan at
``~/.claude/plans/explore-in-depth-agents-from-curried-muffin.md`` for the
broader architecture.
"""

from .graph import build_investment_debate_graph, build_investment_thesis_graph
from .researchers import (
    create_bear_researcher_agent,
    create_bull_researcher_agent,
    create_investment_judge_agent,
)
from .schemas import (
    AnalysisContext,
    InvestmentJudgeOutput,
    InvestmentSignal,
    TraderAction,
    TraderOutput,
)
from .state import InvestmentDebateState, TradingDecisionState
from .trader import create_trader_agent

__all__ = [
    "AnalysisContext",
    "InvestmentDebateState",
    "InvestmentJudgeOutput",
    "InvestmentSignal",
    "TraderAction",
    "TraderOutput",
    "TradingDecisionState",
    "build_investment_debate_graph",
    "build_investment_thesis_graph",
    "create_bear_researcher_agent",
    "create_bull_researcher_agent",
    "create_investment_judge_agent",
    "create_trader_agent",
]
