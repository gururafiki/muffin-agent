"""Store configuration for namespace access control."""

from pydantic import Field

from ...utils.base_config import BaseConfiguration


class StoreConfiguration(BaseConfiguration):
    """Namespace-level access control for LangGraph store operations.

    Agents pass allowed namespace prefixes via
    ``RunnableConfig["configurable"]["store_allowed_namespaces"]``.
    """

    store_allowed_namespaces: list[str] | None = Field(
        default=None,
        description=(
            "Namespace prefixes the agent may access "
            "(e.g. ['cache', 'computed']). None = unrestricted."
        ),
    )
