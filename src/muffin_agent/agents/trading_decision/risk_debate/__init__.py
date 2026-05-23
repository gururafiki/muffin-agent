"""Aggressive / Conservative / Neutral risk-debate nodes."""

from .aggressive_debator import (
    AggressiveDebatorInputState,
    AggressiveDebatorOutputState,
    aggressive_debator_node,
)
from .conservative_debator import (
    ConservativeDebatorInputState,
    ConservativeDebatorOutputState,
    conservative_debator_node,
)
from .neutral_debator import (
    NeutralDebatorInputState,
    NeutralDebatorOutputState,
    neutral_debator_node,
)

__all__ = [
    "AggressiveDebatorInputState",
    "AggressiveDebatorOutputState",
    "ConservativeDebatorInputState",
    "ConservativeDebatorOutputState",
    "NeutralDebatorInputState",
    "NeutralDebatorOutputState",
    "aggressive_debator_node",
    "conservative_debator_node",
    "neutral_debator_node",
]
