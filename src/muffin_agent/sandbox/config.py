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

    opensandbox_use_server_proxy: bool = Field(
        default=True,
        description=(
            "Route sandbox traffic through the OpenSandbox server instead of "
            "connecting to sandbox containers directly. Required whenever the agent "
            "cannot reach sandbox container ports directly — e.g. Docker Swarm "
            "overlay / bridge deployments where the server spawns sandboxes on the "
            "host via docker.sock (the server proxies via the Docker API). Set "
            "OPENSANDBOX_USE_SERVER_PROXY=false only for host/flat-network setups "
            "where direct access is faster and reachable."
        ),
    )
