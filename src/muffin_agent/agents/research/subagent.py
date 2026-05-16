"""CompiledSubAgent factory for embedding research inside another agent.

Returns a ``CompiledSubAgent`` wrapping the research graph, suitable
for passing to ``MuffinAgentBuilder.with_subagents([...])`` in a
parent deep agent (e.g. a thesis-building investment agent that needs
broad web research alongside its specialised data-collection subagents).
"""

from __future__ import annotations

from collections.abc import Sequence

from deepagents import CompiledSubAgent
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool

from .graph import build_research_graph


async def build_research_subagent(
    config: RunnableConfig,  # noqa: ARG001 — held for symmetry with peer factories
    *,
    extra_tools: Sequence[BaseTool] | None = None,
    extra_sources: Sequence[str] | None = None,
    name: str = "deep-research",
) -> CompiledSubAgent:
    """Build the research subagent.

    Args:
        config: Caller's ``RunnableConfig``.  Held for symmetry with
            other muffin subagent factories — the inner graph reads
            its own configuration from each per-invocation ``config``.
        extra_tools: Tools to plug into the researcher (e.g. an
            ArXiv search tool, NewsAPI wrapper, or an internal docs
            retriever).
        extra_sources: Source names to expose to the classifier's
            ``sources_to_use`` enum.  Match the ``source_type`` your
            tools populate so the writer can route citations.
        name: Public name shown to the parent agent's ``task`` tool.
            Default ``"deep-research"``.
    """
    runnable = build_research_graph(
        extra_tools=extra_tools, extra_sources=extra_sources
    )
    return CompiledSubAgent(
        name=name,
        description=(
            "Performs domain-agnostic deep web research on a single "
            "question and returns a cited markdown answer, key findings, "
            "source list, confidence, and suggested follow-ups. "
            "Use for fact-finding, comparisons, how-to guides, summaries, "
            "debates, or factual Q&A over the open web. "
            "Pass the user-facing query directly; optionally include "
            "`chat_history` for coref resolution. Do NOT use for tasks "
            "the parent's own specialised subagents already cover."
        ),
        runnable=runnable,
    )
