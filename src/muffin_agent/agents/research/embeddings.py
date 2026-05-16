"""Embedding-based rerank for evidence chunks (Vane parity).

One public function, ``compute_evidence_relevance``: embed the query +
each chunk via an OpenAI-compatible endpoint, cosine-filter at the
configured threshold, dedup by URL (first-URL-wins; merge content by
keeping the longer body), and return the top-K sorted by relevance.

OpenAI-compatible endpoints supported out of the box:

- OpenAI direct (default — set ``OPENAI_API_KEY``).
- OpenRouter (``base_url=https://openrouter.ai/api/v1``, e.g. for the free
  ``nvidia/llama-nemotron-embed-vl-1b-v2:free`` model).
- vLLM / LM Studio / Ollama / any other OpenAI-compatible server.
"""

from __future__ import annotations

import logging

import numpy as np
from langchain_openai import OpenAIEmbeddings
from pydantic import SecretStr

from .schemas import EvidenceChunk

logger = logging.getLogger(__name__)


def _normalise(vec: np.ndarray) -> np.ndarray:
    """L2-normalise a vector; protect against zero-norm with a small epsilon."""
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / (norm + 1e-9)


def _embed_target(chunk: EvidenceChunk) -> str:
    """Pick the text to embed for *chunk*.

    Prefer the snippet (short, focused); fall back to the start of the
    scraped content when no snippet is available.  Truncating long
    content keeps embedding cost bounded.
    """
    if chunk.snippet:
        return chunk.snippet
    return chunk.content[:500]


def _dedup_by_url(
    scored: list[tuple[float, EvidenceChunk]],
) -> dict[str, EvidenceChunk]:
    """First-URL-wins dedup; content-merge keeps the longer body."""
    by_url: dict[str, EvidenceChunk] = {}
    # Process highest-score first so the per-URL kept score is the max.
    for _, chunk in sorted(scored, key=lambda x: -x[0]):
        existing = by_url.get(chunk.url)
        if existing is None:
            by_url[chunk.url] = chunk
            continue
        # Keep the entry already in the dict (highest score wins); merge
        # the longer content body into it so we don't lose information.
        merged_content = (
            existing.content
            if len(existing.content) >= len(chunk.content)
            else chunk.content
        )
        by_url[chunk.url] = existing.model_copy(update={"content": merged_content})
    return by_url


async def compute_evidence_relevance(
    query: str,
    chunks: list[EvidenceChunk],
    *,
    threshold: float,
    top_k: int,
    embedding_model: str,
    embedding_base_url: str | None = None,
    embedding_api_key: str | None = None,
) -> list[EvidenceChunk]:
    """Embed + cosine-filter + URL-dedup + top-K *chunks* against *query*.

    Args:
        query: Standalone (coref-resolved) research query.
        chunks: Evidence chunks emitted by the researcher.
        threshold: Cosine cutoff; chunks below are dropped.
        top_k: Maximum chunks returned (after dedup).
        embedding_model: OpenAI-compatible model name (e.g.
            ``text-embedding-3-small`` or
            ``nvidia/llama-nemotron-embed-vl-1b-v2:free``).
        embedding_base_url: Optional override for the embeddings endpoint.
            ``None`` → OpenAI default.  Set to
            ``https://openrouter.ai/api/v1`` for OpenRouter.
        embedding_api_key: Optional override for the API key.  ``None``
            → falls back to the ``OPENAI_API_KEY`` env var.

    Returns:
        Chunks above *threshold*, deduped by URL, sorted by relevance
        descending, capped at *top_k*.  Each chunk's ``.relevance`` is
        populated.

    Note:
        On embedding-API failure the function logs and returns an empty
        list rather than raising — the writer node degrades to "no
        evidence retrieved" with confidence < 0.9.
    """
    if not chunks:
        return []

    embedder = OpenAIEmbeddings(
        model=embedding_model,
        base_url=embedding_base_url,
        api_key=SecretStr(embedding_api_key) if embedding_api_key else None,
    )

    texts = [query] + [_embed_target(c) for c in chunks]
    try:
        vectors = await embedder.aembed_documents(texts)
    except Exception:  # noqa: BLE001 — degrade gracefully on embed failure
        logger.exception(
            "Evidence rerank embedding call failed; returning empty result"
        )
        return []

    query_vec = _normalise(np.asarray(vectors[0], dtype=np.float32))
    scored: list[tuple[float, EvidenceChunk]] = []
    for chunk, raw_vec in zip(chunks, vectors[1:]):
        chunk_vec = _normalise(np.asarray(raw_vec, dtype=np.float32))
        similarity = float(query_vec @ chunk_vec)
        if similarity < threshold:
            continue
        scored.append((similarity, chunk.model_copy(update={"relevance": similarity})))

    if not scored:
        return []

    deduped = _dedup_by_url(scored)
    ranked = sorted(
        deduped.values(),
        key=lambda c: -(c.relevance or 0.0),
    )
    return ranked[:top_k]
