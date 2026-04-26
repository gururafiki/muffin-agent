"""Configuration and LLM provider management."""

from typing import Any, Literal

from langchain_core.language_models import BaseChatModel
from pydantic import Field

from .utils.base_config import BaseConfiguration

# NOTE: do NOT default to a `:free` OpenRouter route. Free routes have
# aggressive rate limits, frequent mid-stream chunk-merge bugs, and no
# guaranteed availability — they are unsafe as a production default.
# Override per-deployment via the ``MODEL`` env var (or per-role via
# ``ORCHESTRATOR_MODELS`` / ``COLLECTOR_MODELS`` / ``REASONER_MODELS``).
DEFAULT_MODEL = "openai/gpt-oss-120b"

Role = Literal["orchestrator", "collector", "reasoner"]


class ModelConfiguration(BaseConfiguration):
    """LangGraph-compatible configuration for LLM provider selection.

    Two construction paths:

    * :meth:`get_llm` — single-model legacy path (uses ``llm_provider``).
    * :meth:`get_llm_for_role` — role-aware path. Returns the ordered model
      chain for *role*: the first entry is the primary, the rest are fallbacks.
      Each entry is a bare model name resolved through :meth:`get_llm`, so all
      entries share the configured provider. For mixed-provider chains, set
      ``llm_provider="openrouter"`` and list different OpenRouter model IDs.
      When the role chain is empty, returns a single-element list with the
      legacy default model.
    """

    model: str = Field(default=DEFAULT_MODEL)
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    llm_provider: Literal["openai", "anthropic", "openrouter"] = "openai"
    llm_sdk_retries: int = Field(default=6, ge=0, le=10)

    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    openrouter_api_key: str | None = None
    openai_site_url: str | None = None

    orchestrator_models: list[str] = Field(default_factory=list)
    collector_models: list[str] = Field(default_factory=list)
    reasoner_models: list[str] = Field(default_factory=list)

    def get_llm(
        self, model: str | None = None, temperature: float | None = None, **kwargs: Any
    ) -> BaseChatModel:
        """Get a single LLM instance based on ``llm_provider``."""
        model = model or self.model
        temperature = temperature if temperature is not None else self.temperature

        if self.llm_provider == "openai":
            from langchain_openai import ChatOpenAI

            if not self.openai_api_key:
                raise ValueError("OpenAI API key not configured")
            return ChatOpenAI(
                model=model,
                temperature=temperature,
                api_key=self.openai_api_key,
                base_url=self.openai_site_url,
                default_headers={"HTTP-Referer": self.openai_site_url or ""},
                max_retries=self.llm_sdk_retries,
                **kwargs,
            )
        if self.llm_provider == "anthropic":
            from langchain_anthropic import ChatAnthropic

            if not self.anthropic_api_key:
                raise ValueError("Anthropic API key not configured")
            return ChatAnthropic(
                model=model,
                temperature=temperature,
                api_key=self.anthropic_api_key,
                max_retries=self.llm_sdk_retries,
                **kwargs,
            )
        if self.llm_provider == "openrouter":
            from langchain_openrouter import ChatOpenRouter

            if not self.openrouter_api_key:
                raise ValueError("OpenRouter API key not configured")
            return ChatOpenRouter(
                model=model,
                temperature=temperature,
                openrouter_api_key=self.openrouter_api_key,
                max_retries=self.llm_sdk_retries,
                **kwargs,
            )
        raise ValueError(f"Unsupported LLM provider: {self.llm_provider}")

    def get_llm_for_role(
        self,
        role: Role,
        *,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> list[BaseChatModel]:
        """Return the ordered model chain for *role*.

        First entry is the primary; remainder are fallbacks. Always non-empty —
        if the role chain is unset, returns ``[self.get_llm(...)]``.
        """
        chain = getattr(self, f"{role}_models")
        if not chain:
            return [self.get_llm(temperature=temperature, **kwargs)]
        return [
            self.get_llm(model=entry, temperature=temperature, **kwargs)
            for entry in chain
        ]
