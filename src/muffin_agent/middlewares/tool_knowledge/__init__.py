"""Tool knowledge middleware — adaptive learning from tool failures.

Replaces the older ``ToolErrorHandlerMiddleware``: same duplicate-block
behaviour, plus a lesson recorder. With a summariser model the lesson is
LLM-distilled; without one the middleware falls back to a deterministic
``"<tool>: previous call failed — <error>"`` string so lessons still
accumulate.

Internal layout (kept small per file):

* ``errors.py`` — permanent-vs-transient classifier + duplicate-block key.
* ``lessons.py`` — store-backed CRUD wrapper (``LessonCatalog``).
* ``summariser.py`` — LLM-based summariser + deterministic fallback.
* ``prompt.py`` — render and stitch the system-prompt addendum.
* ``middleware.py`` — wires the four pieces into LangChain hooks.
"""

from .errors import is_permanent_error
from .lessons import Lesson, LessonCatalog, lessons_namespace
from .middleware import ToolKnowledgeMiddleware, ToolLessonState
from .summariser import DEFAULT_SUMMARISER_PROMPT, fallback_lesson

__all__ = [
    "DEFAULT_SUMMARISER_PROMPT",
    "Lesson",
    "LessonCatalog",
    "ToolKnowledgeMiddleware",
    "ToolLessonState",
    "fallback_lesson",
    "is_permanent_error",
    "lessons_namespace",
]
