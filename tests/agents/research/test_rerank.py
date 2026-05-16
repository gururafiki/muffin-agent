"""Tests for the rerank node and the embeddings helper."""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.runnables import RunnableConfig

from muffin_agent.agents.research.embeddings import compute_evidence_relevance
from muffin_agent.agents.research.nodes.rerank import rerank_node
from muffin_agent.agents.research.schemas import EvidenceChunk


@pytest.mark.unit
@pytest.mark.asyncio
class TestComputeEvidenceRelevance:
    async def test_empty_chunks_returns_empty(self):
        result = await compute_evidence_relevance(
            "any query",
            [],
            threshold=0.5,
            top_k=10,
            embedding_model="anything",
        )
        assert result == []

    async def test_threshold_filters_unrelated(
        self,
        evidence_chunks: list[EvidenceChunk],
        mock_embedder: dict[str, Any],  # noqa: ARG002 — fixture activates patch
    ):
        result = await compute_evidence_relevance(
            "How does pgvector indexing work?",
            evidence_chunks,
            threshold=0.5,
            top_k=10,
            embedding_model="text-embedding-3-small",
        )
        urls = [c.url for c in result]
        # pgvector + supabase pass; python tutorial filtered out.
        assert "https://github.com/pgvector/pgvector" in urls
        assert "https://supabase.com/docs/guides/ai/vector-columns" in urls
        assert "https://example.com/python-101" not in urls

    async def test_ranked_descending_by_relevance(
        self,
        evidence_chunks: list[EvidenceChunk],
        mock_embedder: dict[str, Any],  # noqa: ARG002
    ):
        result = await compute_evidence_relevance(
            "How does pgvector indexing work?",
            evidence_chunks,
            threshold=0.0,
            top_k=10,
            embedding_model="text-embedding-3-small",
        )
        relevances = [c.relevance for c in result]
        pairs = zip(relevances, relevances[1:])
        assert all(a is not None and b is not None and a >= b for a, b in pairs)

    async def test_top_k_caps_result(
        self,
        evidence_chunks: list[EvidenceChunk],
        mock_embedder: dict[str, Any],  # noqa: ARG002
    ):
        result = await compute_evidence_relevance(
            "How does pgvector indexing work?",
            evidence_chunks,
            threshold=0.0,
            top_k=1,
            embedding_model="text-embedding-3-small",
        )
        assert len(result) == 1

    async def test_url_dedup_keeps_longer_content(
        self,
        mock_embedder: dict[str, Any],  # noqa: ARG002
    ):
        # Two chunks with the same URL but different content lengths.
        chunks = [
            EvidenceChunk(
                title="pgvector",
                url="https://github.com/pgvector/pgvector",
                snippet="pgvector adds vector similarity search to Postgres.",
                content="short",
            ),
            EvidenceChunk(
                title="pgvector again",
                url="https://github.com/pgvector/pgvector",
                snippet="pgvector adds vector similarity search to Postgres.",
                content="this is a much longer body that should win",
            ),
        ]
        result = await compute_evidence_relevance(
            "How does pgvector indexing work?",
            chunks,
            threshold=0.0,
            top_k=10,
            embedding_model="text-embedding-3-small",
        )
        assert len(result) == 1
        assert "longer body" in result[0].content

    async def test_provider_config_passthrough(
        self,
        evidence_chunks: list[EvidenceChunk],
        mock_embedder: dict[str, Any],
    ):
        await compute_evidence_relevance(
            "How does pgvector indexing work?",
            evidence_chunks,
            threshold=0.5,
            top_k=10,
            embedding_model="nvidia/llama-nemotron-embed-vl-1b-v2:free",
            embedding_base_url="https://openrouter.ai/api/v1",
            embedding_api_key="sk-or-test",
        )
        assert mock_embedder["model"] == "nvidia/llama-nemotron-embed-vl-1b-v2:free"
        assert mock_embedder["base_url"] == "https://openrouter.ai/api/v1"
        # api_key is wrapped in SecretStr before being passed to OpenAIEmbeddings.
        from pydantic import SecretStr

        assert isinstance(mock_embedder["api_key"], SecretStr)
        assert mock_embedder["api_key"].get_secret_value() == "sk-or-test"


@pytest.mark.unit
@pytest.mark.asyncio
class TestRerankNode:
    async def test_empty_evidence_returns_empty_reranked(self):
        state: dict = {"evidence": [], "standalone_query": "x"}
        result = await rerank_node(state, RunnableConfig(configurable={}))
        assert result == {"reranked_evidence": []}

    async def test_dispatches_to_helper(
        self,
        evidence_chunks: list[EvidenceChunk],
        mock_embedder: dict[str, Any],  # noqa: ARG002
    ):
        state: dict = {
            "evidence": [c.model_dump() for c in evidence_chunks],
            "standalone_query": "How does pgvector indexing work?",
        }
        result = await rerank_node(state, RunnableConfig(configurable={}))
        reranked = result["reranked_evidence"]
        urls = [c["url"] for c in reranked]
        # Same expectations as the helper test.
        assert "https://github.com/pgvector/pgvector" in urls
        assert "https://example.com/python-101" not in urls

    async def test_malformed_evidence_returns_empty(self):
        state: dict = {
            "evidence": [{"this_is_not": "a valid evidence chunk"}],
            "standalone_query": "x",
        }
        result = await rerank_node(state, RunnableConfig(configurable={}))
        assert result == {"reranked_evidence": []}
