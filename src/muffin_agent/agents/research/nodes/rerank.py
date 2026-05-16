"""Rerank node: embedding-based cosine filter on collected evidence.

Pure Python.  No LLM.  Calls ``compute_evidence_relevance`` (which
talks to an OpenAI-compatible embeddings endpoint) and writes the
filtered + ranked list back to state.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.runnables import RunnableConfig

from ..config import ResearchConfiguration
from ..embeddings import compute_evidence_relevance
from ..schemas import EvidenceChunk
from ..state import ResearchState

logger = logging.getLogger(__name__)


async def rerank_node(
    state: ResearchState,
    config: RunnableConfig,
) -> dict[str, Any]:
    """Cosine-rank ``state["evidence"]`` and write ``reranked_evidence``."""
    evidence_dicts = state.get("evidence") or []
    if not evidence_dicts:
        return {"reranked_evidence": []}

    research_cfg = ResearchConfiguration.from_runnable_config(config)
    query = state.get("standalone_query") or state.get("query", "")

    try:
        chunks = [EvidenceChunk(**c) for c in evidence_dicts]
    except Exception:  # noqa: BLE001 — malformed chunk should not kill the pipeline
        logger.exception("rerank_node: failed to parse evidence; returning empty")
        return {"reranked_evidence": []}

    ranked = await compute_evidence_relevance(
        query=query,
        chunks=chunks,
        threshold=research_cfg.rerank_threshold,
        top_k=research_cfg.rerank_top_k,
        embedding_model=research_cfg.embedding_model,
        embedding_base_url=research_cfg.embedding_base_url,
        embedding_api_key=research_cfg.embedding_api_key,
    )

    return {"reranked_evidence": [c.model_dump() for c in ranked]}
