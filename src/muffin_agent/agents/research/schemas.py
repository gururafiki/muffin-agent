"""Pydantic schemas for the research agent.

Wire shapes:

- ``ResearchClassification`` — classifier node's structured output.
- ``EvidenceChunk`` — one piece of retrieved evidence.
- ``ResearchEvidenceFindings`` — researcher node's structured output
  (list of evidence chunks plus optional free-form notes).
- ``Source`` — citation entry exposed in the final answer.
- ``ResearchOutput`` — writer node's structured output; this is the
  agent's public contract.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ResearchClassification(BaseModel):
    """Classifier output.  Drives every downstream stage."""

    standalone_query: str = Field(
        description=(
            "Self-contained version of the user's query with all "
            'co-reference ("this", "that", "they") resolved against '
            "chat history.  Researcher and writer never see the raw query."
        ),
    )
    task_type: Literal[
        "research_report",
        "comparison",
        "how_to",
        "summary",
        "debate",
        "factual_qa",
    ] = Field(description="Shape of the final answer.")
    mode_hint: Literal["speed", "balanced", "quality"] = Field(
        description=(
            "Research depth.  Caller can override via mode_override; "
            "otherwise this is honoured."
        ),
    )
    sources_to_use: list[str] = Field(
        description=(
            "Subset of allowed_sources the researcher should consult.  "
            "Always a subset — never include sources outside allowed_sources."
        ),
    )
    skip_search: bool = Field(
        description=(
            "True for purely chit-chat or arithmetic queries that don't "
            "need any external lookup.  Writer answers from internal "
            "knowledge with confidence < 0.9 and empty sources."
        ),
    )
    rationale: str = Field(
        default="",
        description="One-sentence why; helpful for tracing, not user-facing.",
    )


class EvidenceChunk(BaseModel):
    """One piece of retrieved evidence.

    Researcher emits these via its structured response.  Rerank node
    embeds + cosine-filters them.  Writer consumes the reranked set.
    """

    title: str
    url: str
    snippet: str = Field(
        description="Short summary (typically search-result snippet).",
    )
    content: str = Field(
        default="",
        description=(
            "Full scraped content (markdown) when the researcher chose to "
            "scrape this URL.  Empty when only the snippet is available."
        ),
    )
    source_type: str = Field(default="web", description="web/academic/news/…")
    retrieved_at: str | None = Field(
        default=None, description="ISO-8601 timestamp when retrieved."
    )
    relevance: float | None = Field(
        default=None,
        description="Cosine similarity vs the standalone query; set by rerank.",
        ge=0.0,
        le=1.0,
    )


class ResearchEvidenceFindings(BaseModel):
    """Researcher node's structured output.

    The deep agent emits this as ``structured_response`` once it decides
    it has gathered enough evidence (or hits the iteration cap).  Free-
    form messages from the researcher are NOT consumed downstream — the
    rerank and writer stages read ``evidence_chunks`` only.
    """

    evidence_chunks: list[EvidenceChunk] = Field(
        description="All retrieved evidence the researcher believes is relevant.",
    )
    notes: str = Field(
        default="",
        description=(
            "Optional free-form research notes (e.g. gaps the researcher "
            "couldn't fill).  Not shown to the writer; surfaced in traces."
        ),
    )


class Source(BaseModel):
    """One citation slot in the final answer."""

    n: int = Field(description="1-indexed citation number; matches inline [N].")
    title: str
    url: str


class ResearchOutput(BaseModel):
    """Writer node's structured output — the agent's public contract."""

    answer_markdown: str = Field(
        description=(
            "Final answer as markdown.  Every non-trivial claim must "
            "carry an inline [N] citation matching an entry in `sources`."
        ),
    )
    key_findings: list[str] = Field(
        description="3-7 bullet points distilling the answer.",
    )
    sources: list[Source] = Field(
        description="Citation list keyed by `n` (matches inline [N] markers).",
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Overall confidence in the answer.  <0.9 when no sources were "
            "retrieved (skip_search path) or evidence is contradictory."
        ),
    )
    missing_information: list[str] = Field(
        default_factory=list,
        description="Specific facts/data the agent could not find.",
    )
    suggested_followups: list[str] = Field(
        default_factory=list,
        description="3-5 plausible follow-up queries.",
    )
    task_type: str = Field(description="Echoed from classifier.")
    mode_used: str = Field(description="Echoed from classifier (or override).")
