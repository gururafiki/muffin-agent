"""Configuration and LLM provider management."""

import os
from typing import Any, Literal

from dotenv import load_dotenv
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

load_dotenv()

DEFAULT_MODEL = "upstage/solar-pro-3:free"


class Configuration(BaseModel):
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

    # ==================== Criteria Selection ====================

    max_criteria: int = Field(
        default=7,
        ge=1,
        le=20,
        description="Maximum number of evaluation criteria for the reasoning agent",
    )

    # ==================== MCP Servers ====================

    def get_mcp_connections(self) -> dict[str, dict]:
        """Get MCP server connections for MultiServerMCPClient."""
        return {
            "openbb": {
                "url": "http://127.0.0.1:8001/mcp",
                "transport": "streamable_http",
            }
        }

    @classmethod
    def from_runnable_config(cls, config: RunnableConfig) -> "Configuration":
        """Create Configuration from a LangGraph RunnableConfig.

        Extracts known fields from config["configurable"], ignoring unknown keys.
        """
        configurable = config.get("configurable", {})

        # Get raw values from environment or config
        raw_values: dict[str, Any] = {
            name: os.environ.get(name.upper(), configurable.get(name))
            for name in cls.model_fields.keys()
        }
        # Filter out None values
        values = {k: v for k, v in raw_values.items() if v is not None}

        return cls(**values)


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
                **kwargs,
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {self.llm_provider}")
