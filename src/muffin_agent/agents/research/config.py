"""Configuration for the deep research agent.

Knobs for embedding provider, rerank thresholds, default behaviour, and
per-mode iteration caps.  Reads env vars via the inherited
``BaseConfiguration.from_runnable_config``.
"""

from typing import Literal

from pydantic import Field

from ...utils.base_config import BaseConfiguration


class ResearchConfiguration(BaseConfiguration):
    """Research-agent knobs read from env / RunnableConfig."""

    # ── Embedding provider (OpenAI-compatible) ──────────────────────────
    # Defaults target OpenAI direct.  Override ``embedding_base_url`` to
    # point at any OpenAI-compatible endpoint (OpenRouter, vLLM, LM Studio,
    # Ollama).  ``embedding_api_key`` falls back to ``OPENAI_API_KEY`` env
    # var when unset — convenient for the existing OpenRouter pattern
    # (``OPENAI_API_KEY=<openrouter-key>`` + ``OPENAI_SITE_URL=...``).
    embedding_model: str = "text-embedding-3-small"
    embedding_base_url: str | None = None
    embedding_api_key: str | None = None

    # ── Rerank (Vane defaults) ──────────────────────────────────────────
    rerank_threshold: float = 0.5
    rerank_top_k: int = 20

    # ── Behaviour defaults ──────────────────────────────────────────────
    research_default_mode: Literal["speed", "balanced", "quality"] = "balanced"
    research_default_sources: list[str] = Field(default_factory=lambda: ["web"])
    max_search_results: int = 8

    # ── Iteration caps per mode (Vane parity) ───────────────────────────
    research_iter_speed: int = 2
    research_iter_balanced: int = 6
    research_iter_quality: int = 25

    def iter_budget_for(self, mode: str) -> int:
        """Return the LLM-call budget for *mode*; falls back to balanced."""
        return {
            "speed": self.research_iter_speed,
            "balanced": self.research_iter_balanced,
            "quality": self.research_iter_quality,
        }.get(mode, self.research_iter_balanced)
