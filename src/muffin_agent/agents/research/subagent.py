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
    config: RunnableConfig,
    *,
    extra_tools: Sequence[BaseTool] | None = None,
    name: str = "deep-research",
) -> CompiledSubAgent:
    """Build the research subagent.

    Args:
        config: Caller's ``RunnableConfig``.  Now genuinely used — the inner graph
            builds its compiled agent nodes at construction time and needs the
            model / MCP configuration to do so.
        extra_tools: Tools to plug into the researcher (e.g. an
            ArXiv search tool, NewsAPI wrapper, or an internal docs
            retriever).
        name: Public name shown to the parent agent's ``task`` tool.
            Default ``"deep-research"``.

    Note:
        ``extra_sources`` was removed — the permitted source list is now per-run
        state (``allowed_sources``) rather than baked into the graph, so callers
        pass it in the invocation input instead.
    """
    runnable = await build_research_graph(config, extra_tools=extra_tools)
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
