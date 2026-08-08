"""Base configuration shared by all Pydantic config classes."""

import os
import typing
from typing import Any, Self

from dotenv import load_dotenv
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel

load_dotenv()


def _is_list_field(annotation: Any) -> bool:
    """Detect ``list[...]`` (and ``list[...] | None``) field annotations."""
    if typing.get_origin(annotation) is list:
        return True
    union_types = (typing.Union, getattr(typing, "UnionType", None))
    if typing.get_origin(annotation) in union_types:
        args = typing.get_args(annotation)
        return any(typing.get_origin(arg) is list for arg in args)
    return False


class BaseConfiguration(BaseModel):
    """Base for all muffin-agent configuration models."""

    @classmethod
    def from_runnable_config(cls, config: RunnableConfig) -> Self:
        """Create Configuration from a LangGraph RunnableConfig.

        Extracts known fields from config["configurable"], ignoring unknown
        keys. Comma-separated env-var strings populate ``list[...]`` fields; a
        JSON-array string (``[...]``, e.g. ``LLM_CHAIN``) is passed through
        verbatim for the field's own validator to parse.

        The ``-> Self`` annotation is load-bearing, not decoration. Without it
        this returned ``Any``, and ``Any`` flows silently into anything: 16 CLI
        commands passed the *result* of this call into agent factories that take
        a ``RunnableConfig``, and every one of them died on
        ``AttributeError: 'ModelConfiguration' object has no attribute 'get'``
        the moment the factory called this method back on it. mypy could only see
        the two that also misspelled the factory name. This is the same defect
        class CLAUDE.md records for the retired ``run_deep_agent_node`` wrapper —
        keep the annotation so it stays visible.
        """
        configurable = config.get("configurable", {})

        values: dict[str, Any] = {}
        for name, field in cls.model_fields.items():
            raw = os.environ.get(name.upper(), configurable.get(name))
            if raw is None:
                continue
            if isinstance(raw, str) and _is_list_field(field.annotation):
                stripped = raw.strip()
                # A JSON array is left for the field's before-validator (e.g.
                # llm_chain); only simple "a,b,c" strings are comma-split here.
                raw = (
                    stripped
                    if stripped.startswith("[")
                    else [item.strip() for item in raw.split(",") if item.strip()]
                )
            values[name] = raw

        return cls(**values)
