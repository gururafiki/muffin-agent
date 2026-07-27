"""Node implementations for the research pipeline.

The three LLM stages are exposed as **agent factories**, not node functions — each is
added to the graph as a compiled agent via ``add_node`` so it owns its own
``checkpoint_ns``. The remaining exports are the pure-Python nodes.
"""

from .classifier import (
    create_classifier_agent,
    lift_classification_node,
    prepare_node,
)
from .rerank import rerank_node
from .researcher import build_researchers_by_mode, create_researcher_agent
from .writer import create_writer_agent, finalize_output_node

__all__ = [
    "build_researchers_by_mode",
    "create_classifier_agent",
    "create_researcher_agent",
    "create_writer_agent",
    "finalize_output_node",
    "lift_classification_node",
    "prepare_node",
    "rerank_node",
]
