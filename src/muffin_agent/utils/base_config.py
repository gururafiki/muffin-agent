"""Base configuration shared by all Pydantic config classes."""

import os
from typing import Any

from dotenv import load_dotenv
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel

load_dotenv()


class BaseConfiguration(BaseModel):
    """Base for all muffin-agent configuration models."""

    @classmethod
    def from_runnable_config(cls, config: RunnableConfig):
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
