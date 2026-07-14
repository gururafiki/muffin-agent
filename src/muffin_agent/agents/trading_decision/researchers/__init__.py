"""Investment Judge node for the trading_decision graphs.

The Bull/Bear debate itself has migrated to the multi_agent conference
framework (built inline in :mod:`..graph` as the ``investment_debate``
subgraph); the Judge stays a plain parent-graph node that synthesises the
completed ``investment_debate_messages`` transcript.
"""

from .investment_judge import (
    InvestmentJudgeInputState,
    InvestmentJudgeOutputState,
    investment_judge_node,
)

__all__ = [
    "InvestmentJudgeInputState",
    "InvestmentJudgeOutputState",
    "investment_judge_node",
]
