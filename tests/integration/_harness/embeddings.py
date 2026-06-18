"""The embeddings seam — a canned-vector fake for the research rerank step.

``agents/research/rerank_node`` embeds the query + each evidence chunk via
``langchain_openai.OpenAIEmbeddings`` (imported into
``muffin_agent.agents.research.embeddings``). This fake returns deterministic
vectors so the cosine-filter / dedup / top-K math runs for real without a network
call. Mirrors the monkeypatch already used in ``tests/agents/research/conftest.py``.

Texts not present in *vectors* get a zero-ish default vector (low similarity), so
authoring only needs to pin the handful of texts a test cares about.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator
from unittest.mock import patch

_DEFAULT_DIM = 8


@contextmanager
def patch_embeddings(
    vectors: dict[str, list[float]] | None = None,
) -> Iterator[dict[str, list[float]]]:
    """Patch research ``OpenAIEmbeddings`` to return canned vectors."""
    table = dict(vectors or {})

    class _FakeEmbeddings:
        def __init__(
            self, *, model: str = "", base_url=None, api_key=None, **_: object
        ):  # noqa: ANN001
            self.model = model
            self.base_url = base_url

        def _vec(self, text: str) -> list[float]:
            return table.get(text, [0.0] * _DEFAULT_DIM)

        def embed_query(self, text: str) -> list[float]:
            return self._vec(text)

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return [self._vec(t) for t in texts]

        async def aembed_query(self, text: str) -> list[float]:
            return self._vec(text)

        async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
            return [self._vec(t) for t in texts]

    with patch(
        "muffin_agent.agents.research.embeddings.OpenAIEmbeddings", _FakeEmbeddings
    ):
        yield table
