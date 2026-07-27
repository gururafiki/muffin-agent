"""Records what an agent really ran, and can refuse a data-free answer.

The backstop for the worst observed failure mode: a weak model single-shotting
its structured output with zero tool calls and fabricating the evidence. See
``middleware.py``.
"""

from .middleware import (
    DataCollectionGuardMiddleware,
    DataCollectionGuardState,
    executed_tool_labels,
)

__all__ = [
    "DataCollectionGuardMiddleware",
    "DataCollectionGuardState",
    "executed_tool_labels",
]
