"""General-purpose deep research agent (domain-agnostic).

Public entrypoints:

- :func:`build_research_graph` — compile the LangGraph pipeline
  (prepare → classifier → lift → researcher → rerank → writer →
  finalize).  **Async** — every LLM stage is a compiled agent built at
  graph-construction time.
- :func:`make_graph` — the config-only Platform factory registered in
  ``langgraph.json``; also what the ``muffin research`` CLI drives.
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
from .graph import build_research_graph, make_graph
from .schemas import (
    EvidenceChunk,
    ResearchClassification,
    ResearcherNodeOutput,
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
    "ResearcherNodeOutput",
    "ResearchMode",
    "ResearchOutput",
    "ResearchState",
    "Source",
    "TaskType",
    "build_research_graph",
    "build_research_subagent",
    "make_graph",
]
