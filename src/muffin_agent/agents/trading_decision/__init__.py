"""Trading decision pipeline — composable building blocks ported from TradingAgents.

PR 1 surface (this release):

* :func:`build_investment_debate_graph` — Bull ↔ Bear debate → Investment Judge
* :class:`AnalysisContext` — generic envelope for upstream analysis context
* :class:`InvestmentJudgeOutput` — structured judge output
* :class:`TradingDecisionState` / :class:`InvestmentDebateState` — state schemas
* Standalone agent factories: ``create_{bull_researcher,bear_researcher,
  investment_judge}_agent``

Later PRs add the Trader, Risk Debate, Portfolio Manager, and reflection
memory layers without changing the PR 1 surface.

See ``CLAUDE.md`` and the plan at
``~/.claude/plans/explore-in-depth-agents-from-curried-muffin.md`` for the
broader architecture.
"""

from .graph import build_investment_debate_graph
from .researchers import (
    create_bear_researcher_agent,
    create_bull_researcher_agent,
    create_investment_judge_agent,
)
from .schemas import AnalysisContext, InvestmentJudgeOutput, InvestmentSignal
from .state import InvestmentDebateState, TradingDecisionState

__all__ = [
    "AnalysisContext",
    "InvestmentDebateState",
    "InvestmentJudgeOutput",
    "InvestmentSignal",
    "TradingDecisionState",
    "build_investment_debate_graph",
    "create_bear_researcher_agent",
    "create_bull_researcher_agent",
    "create_investment_judge_agent",
]
