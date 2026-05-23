"""Bull / Bear / Judge researcher nodes for the trading_decision graphs."""

from .bear_researcher import (
    BearResearcherInputState,
    BearResearcherOutputState,
    bear_researcher_node,
)
from .bull_researcher import (
    BullResearcherInputState,
    BullResearcherOutputState,
    bull_researcher_node,
)
from .investment_judge import (
    InvestmentJudgeInputState,
    InvestmentJudgeOutputState,
    investment_judge_node,
)

__all__ = [
    "BearResearcherInputState",
    "BearResearcherOutputState",
    "BullResearcherInputState",
    "BullResearcherOutputState",
    "InvestmentJudgeInputState",
    "InvestmentJudgeOutputState",
    "bear_researcher_node",
    "bull_researcher_node",
    "investment_judge_node",
]
