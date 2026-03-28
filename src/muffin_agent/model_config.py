"""Configuration and LLM provider management."""

from typing import Any, Literal

from pydantic import Field

from .utils.base_config import BaseConfiguration

DEFAULT_MODEL = "openai/gpt-oss-120b:free"


class ModelConfiguration(BaseConfiguration):
    """LangGraph-compatible configuration for LLM provider selection.

    Model string format: "provider/model-name" where provider is one of:
    openai, anthropic, openrouter.

    For OpenRouter models, use "openrouter/provider/model" format
    (e.g. "openrouter/google/gemini-2.0-flash-001").

    Usage:
        configuration = Configuration.from_runnable_config(config)
        llm = configuration.get_llm(temperature=0.1)
    """

    # ==================== Core Settings ====================

    model: str = Field(
        default=DEFAULT_MODEL,
        description="Default model name (provider-specific)",
    )

    temperature: float = Field(
        default=0.1,
        ge=0.0,
        le=2.0,
        description="LLM temperature for analysis tasks (lower = more deterministic)",
    )

    llm_provider: Literal["openai", "anthropic"] = Field(
        default="openai",
        description="Default LLM provider to use",
    )

    llm_max_retries: int = Field(
        default=6,
        ge=0,
        le=10,
        description=(
            "Retries for transient LLM errors (e.g. HTTP 429 rate limits). "
            "Both ChatOpenAI and ChatAnthropic use exponential backoff."
        ),
    )

    # ==================== API Keys ====================

    openai_api_key: str | None = Field(
        default=None,
        description="OpenAI API key",
    )

    anthropic_api_key: str | None = Field(
        default=None,
        description="Anthropic API key",
    )

    openai_site_url: str | None = Field(
        default=None,
        description="OpenRouter site URL (for backward compatibility)",
    )

    def get_llm(
        self, model: str | None = None, temperature: float | None = None, **kwargs
    ) -> Any:
        """Get LLM instance based on configuration.

        Args:
            model: Override default model
            temperature: Override default temperature
            **kwargs: Additional arguments to pass to LLM constructor

        Returns:
            Initialized LLM instance (ChatOpenAI, ChatAnthropic, etc.)

        Raises:
            ValueError: If provider not supported or API key missing
        """
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
                max_retries=self.llm_max_retries,
                **kwargs,
            )
        elif self.llm_provider == "anthropic":
            from langchain_anthropic import ChatAnthropic

            if not self.anthropic_api_key:
                raise ValueError("Anthropic API key not configured")

            return ChatAnthropic(
                model=model,
                temperature=temperature,
                api_key=self.anthropic_api_key,
                max_retries=self.llm_max_retries,
                **kwargs,
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {self.llm_provider}")
