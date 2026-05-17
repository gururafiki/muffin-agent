"""Aggressive / Conservative / Neutral risk-debate agents."""

from .aggressive_debator import create_aggressive_debator_agent
from .conservative_debator import create_conservative_debator_agent
from .neutral_debator import create_neutral_debator_agent

__all__ = [
    "create_aggressive_debator_agent",
    "create_conservative_debator_agent",
    "create_neutral_debator_agent",
]
