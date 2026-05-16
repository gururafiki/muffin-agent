"""Shared fixtures for research-agent tests."""

from __future__ import annotations

from typing import Any

import pytest

from muffin_agent.agents.research.schemas import (
    EvidenceChunk,
    ResearchClassification,
    ResearchEvidenceFindings,
    ResearchOutput,
)


@pytest.fixture
def sample_classification() -> ResearchClassification:
    """A canned classification result for ``"How does pgvector indexing work?"``."""
    return ResearchClassification(
        standalone_query="How does pgvector indexing work?",
        task_type="how_to",
        mode_hint="balanced",
        sources_to_use=["web"],
        skip_search=False,
        rationale="how-to query about a specific technical mechanism",
    )


@pytest.fixture
def evidence_chunks() -> list[EvidenceChunk]:
    """A diverse set of evidence chunks for testing rerank."""
    return [
        EvidenceChunk(
            title="pgvector README",
            url="https://github.com/pgvector/pgvector",
            snippet="pgvector adds vector similarity search to Postgres.",
            content="pgvector supports IVFFlat and HNSW indexes...",
            source_type="web",
        ),
        EvidenceChunk(
            title="Supabase docs — pgvector",
            url="https://supabase.com/docs/guides/ai/vector-columns",
            snippet="pgvector enables AI features on Supabase.",
            content="To create an HNSW index: CREATE INDEX ...",
            source_type="web",
        ),
        EvidenceChunk(
            title="Unrelated python tutorial",
            url="https://example.com/python-101",
            snippet="An introduction to Python programming.",
            content="Python is a high-level language.",
            source_type="web",
        ),
    ]


@pytest.fixture
def sample_evidence_findings(
    evidence_chunks: list[EvidenceChunk],
) -> ResearchEvidenceFindings:
    return ResearchEvidenceFindings(
        evidence_chunks=evidence_chunks,
        notes="",
    )


@pytest.fixture
def sample_research_output() -> ResearchOutput:
    """A canned writer output."""
    from muffin_agent.agents.research.schemas import Source

    return ResearchOutput(
        answer_markdown=(
            "pgvector supports two index types: IVFFlat and HNSW[1].\n\n"
            "Supabase ships pgvector with managed Postgres[2]."
        ),
        key_findings=[
            "IVFFlat splits vectors into clusters[1].",
            "HNSW builds a multi-layer graph[1].",
        ],
        sources=[
            Source(
                n=1, title="pgvector README", url="https://github.com/pgvector/pgvector"
            ),
            Source(
                n=2,
                title="Supabase docs",
                url="https://supabase.com/docs/guides/ai/vector-columns",
            ),
        ],
        confidence=0.85,
        missing_information=[],
        suggested_followups=[
            "How do I tune HNSW ef_construction?",
            "What are the trade-offs of IVFFlat vs HNSW?",
        ],
        task_type="how_to",
        mode_used="balanced",
    )


@pytest.fixture
def canned_embeddings() -> dict[str, list[float]]:
    """Tiny synthetic embeddings keyed by the text being embedded.

    Vectors are crafted so that:
    - "How does pgvector indexing work?" is close to the two pgvector
      chunks and far from the unrelated python tutorial.
    """
    return {
        "How does pgvector indexing work?": [1.0, 0.0, 0.0],
        "pgvector adds vector similarity search to Postgres.": [0.9, 0.1, 0.0],
        "pgvector enables AI features on Supabase.": [0.85, 0.05, 0.0],
        "An introduction to Python programming.": [0.0, 0.0, 1.0],
    }


@pytest.fixture
def mock_embedder(monkeypatch: Any, canned_embeddings: dict[str, list[float]]):
    """Patch ``OpenAIEmbeddings`` to return synthetic vectors.

    The fake embedder also records every (model, base_url, api_key)
    constructor call so tests can assert on provider config passthrough.
    """
    recorded: dict[str, Any] = {}

    class _FakeEmbeddings:
        def __init__(
            self,
            *,
            model: str,
            base_url: str | None = None,
            api_key: str | None = None,
        ) -> None:
            recorded["model"] = model
            recorded["base_url"] = base_url
            recorded["api_key"] = api_key

        async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
            return [canned_embeddings.get(t, [0.0, 0.0, 0.0]) for t in texts]

    monkeypatch.setattr(
        "muffin_agent.agents.research.embeddings.OpenAIEmbeddings",
        _FakeEmbeddings,
    )
    return recorded
