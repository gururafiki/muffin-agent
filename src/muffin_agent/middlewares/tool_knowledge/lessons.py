"""Lesson catalog — store-backed CRUD for tool-failure lessons.

Encapsulates the store schema (``("tool_lessons", tool_name)`` namespace,
``error_class_hash`` keys) and the dedup-on-write semantics so the rest
of the middleware doesn't need to know any of it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from .summariser import error_class_hash

logger = logging.getLogger(__name__)

_LESSONS_NAMESPACE_ROOT = "tool_lessons"


def lessons_namespace(tool_name: str) -> tuple[str, ...]:
    """Return the store namespace for *tool_name*'s lessons."""
    return (_LESSONS_NAMESPACE_ROOT, tool_name)


@dataclass(frozen=True)
class Lesson:
    """One stored lesson, in the shape used by the prompt renderer."""

    tool_name: str
    text: str
    created_at: str


class LessonCatalog:
    """Thin wrapper over a ``BaseStore`` for tool-lesson reads & writes."""

    def __init__(self, store: Any) -> None:
        """Bind the catalog to *store* (typically ``runtime.store``)."""
        self._store = store

    async def has(self, tool_name: str, error_message: str) -> bool:
        """Return True when this ``(tool, error_class)`` was already recorded."""
        if self._store is None:
            return False
        ns = lessons_namespace(tool_name)
        key = error_class_hash(tool_name, error_message)
        try:
            return (await self._store.aget(ns, key)) is not None
        except Exception:
            logger.debug("Lesson store read failed for %s", tool_name, exc_info=True)
            return False

    async def record(
        self,
        *,
        tool_name: str,
        args: dict[str, Any],
        error_message: str,
        lesson: str,
    ) -> None:
        """Persist a lesson, swallowing store-side failures."""
        if self._store is None:
            return
        ns = lessons_namespace(tool_name)
        key = error_class_hash(tool_name, error_message)
        try:
            await self._store.aput(
                ns,
                key,
                {
                    "tool_name": tool_name,
                    "lesson": lesson,
                    "error_excerpt": error_message[:240],
                    "args_sample": args,
                    "created_at": datetime.now(UTC).isoformat(),
                },
            )
        except Exception:
            logger.debug(
                "Lesson store write failed for %s", tool_name, exc_info=True
            )

    async def latest_per_tool(
        self,
        tool_names: list[str],
        *,
        cap: int,
    ) -> list[Lesson]:
        """Fetch the *cap* newest lessons across the given tool names."""
        if self._store is None or not tool_names:
            return []
        results: list[Lesson] = []
        for tool_name in tool_names:
            try:
                items = await self._store.asearch(lessons_namespace(tool_name))
            except Exception:
                logger.debug(
                    "Lesson store search failed for %s", tool_name, exc_info=True
                )
                continue
            ordered = sorted(
                items,
                key=lambda it: it.value.get("created_at", ""),
                reverse=True,
            )[:cap]
            for item in ordered:
                text = item.value.get("lesson")
                if isinstance(text, str) and text.strip():
                    results.append(
                        Lesson(
                            tool_name=tool_name,
                            text=text.strip(),
                            created_at=item.value.get("created_at", ""),
                        )
                    )
        return results
