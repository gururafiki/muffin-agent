"""Bull / Bear / Judge researchers for the trading_decision debate."""

from .bear_researcher import create_bear_researcher_agent
from .bull_researcher import create_bull_researcher_agent
from .investment_judge import create_investment_judge_agent

__all__ = [
    "create_bear_researcher_agent",
    "create_bull_researcher_agent",
    "create_investment_judge_agent",
]
