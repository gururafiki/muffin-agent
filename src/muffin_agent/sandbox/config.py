"""Configuration and LLM provider management."""

from pydantic import Field

from ..utils.base_config import BaseConfiguration


class OpenSandboxConfiguration(BaseConfiguration):
    """OpenSandbox container configuration."""

    opensandbox_url: str = Field(
        default="localhost:8080",
        description=(
            "OpenSandbox server address (host:port). "
            "Used as ConnectionConfig.domain when creating sandbox containers."
        ),
    )

    opensandbox_api_key: str | None = Field(
        default=None,
        description="OpenSandbox server API key (leave empty if server has no auth).",
    )

    opensandbox_image: str = Field(
        default="python:3.11-slim",
        description=(
            "Docker image for sandbox containers. "
            "Override with a custom image that has pandas, numpy, ta-lib, etc. "
            "pre-installed for faster startup."
        ),
    )
