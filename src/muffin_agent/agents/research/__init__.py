"""General-purpose deep research agent (domain-agnostic).

Public entrypoints:

- :func:`build_research_graph` — compile the LangGraph pipeline
  (classifier → researcher → rerank → writer).  Registered in
  ``langgraph.json`` and exposed as the ``muffin research`` CLI.
- :func:`build_research_subagent` — wrap the same pipeline as a
  :class:`deepagents.CompiledSubAgent` for embedding inside another
  deep agent.
- :class:`ResearchConfiguration` — runtime knobs (env vars +
  RunnableConfig.configurable).
- :class:`ResearchOutput` — the agent's public output contract.

Pluggability: callers pass ``extra_tools=`` (additional LangChain
``BaseTool`` instances — e.g. an ArXiv search tool, NewsAPI wrapper,
or a finance MCP tool) and ``extra_sources=`` (source names to add
to the classifier's enum).  See ``docs/features/research-agent.md``.
"""

from .config import ResearchConfiguration
from .graph import build_research_graph, graph
from .schemas import (
    EvidenceChunk,
    ResearchClassification,
    ResearchEvidenceFindings,
    ResearchOutput,
    Source,
)
from .state import (
    ResearchClassificationFilterState,
    ResearchMode,
    ResearchState,
    TaskType,
)
from .subagent import build_research_subagent

__all__ = [
    "EvidenceChunk",
    "ResearchClassification",
    "ResearchClassificationFilterState",
    "ResearchConfiguration",
    "ResearchEvidenceFindings",
    "ResearchMode",
    "ResearchOutput",
    "ResearchState",
    "Source",
    "TaskType",
    "build_research_graph",
    "build_research_subagent",
    "graph",
]
