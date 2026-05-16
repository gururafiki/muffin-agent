"""SkillFilterMiddleware — schema-driven skill filtering via flat state keys.

Filters ``skills_metadata`` in ``abefore_agent`` based on classification fields
read from flat state keys, and injects classification context into the system
prompt via ``awrap_model_call``.  Works alongside the default
``SkillsMiddleware`` (via ``skills=`` parameter) which handles skill parsing.

Parameterised via ``__class_getitem__``::

    class TickerClassification(AgentState):
        sector: NotRequired[str]
        market: NotRequired[str]

    middleware = [SkillFilterMiddleware[TickerClassification](
        context_header="Ticker Classification",
        context_intro="This ticker has been classified as follows:",
    )]

The three ``context_*`` constructor kwargs control the prompt section
appended on every model call.  Defaults are domain-agnostic so the same
middleware works for ticker classification, research-mode/task-type
filtering, or any other categorisation scheme.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from deepagents.middleware._utils import append_to_system_message
from deepagents.middleware.skills import SkillMetadata
from langchain.agents import AgentState
from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.tools import BaseTool

if TYPE_CHECKING:
    from langchain.agents.middleware.types import ModelRequest, ModelResponse


class SkillFilterMiddleware(AgentMiddleware):
    """Schema-driven skill filtering middleware.

    Parameterised with an ``AgentState`` subclass whose extra fields (beyond
    ``AgentState``'s own) are the category keys used for filtering.  Uses
    ``__class_getitem__`` so no constructor arguments are needed::

        SkillFilterMiddleware[TickerClassification]()

    The middleware reads flat state keys (not a nested ``classification``
    dict), making it safe to compose with other middlewares that add their
    own state fields.

    Supports multiple skill directories with different or overlapping
    metadata keys — skills whose metadata keys are absent from the
    classification are naturally excluded.
    """

    tools: list[BaseTool] = []
    _category_keys: frozenset[str] = frozenset()

    _DEFAULT_HEADER = "Classification"
    _DEFAULT_INTRO = "The current context has been classified as follows:"
    _DEFAULT_OUTRO = (
        "The available skills listed above have been pre-filtered "
        "to match this classification. Read all of them via `read_file`."
    )

    def __init__(
        self,
        *,
        context_header: str | None = None,
        context_intro: str | None = None,
        context_outro: str | None = None,
    ) -> None:
        """Configure the prompt section appended on every model call.

        Args:
            context_header: Markdown heading text (rendered as ``## <header>``).
                Defaults to ``"Classification"``.
            context_intro: Sentence introducing the bulleted classification
                values.  Defaults to a generic phrasing.
            context_outro: Closing sentence reminding the agent to read the
                pre-filtered skills.  Defaults to a generic phrasing.
        """
        super().__init__()
        self._context_header = context_header or self._DEFAULT_HEADER
        self._context_intro = context_intro or self._DEFAULT_INTRO
        self._context_outro = context_outro or self._DEFAULT_OUTRO

    @classmethod
    def __class_getitem__(cls, filter_schema: type) -> type:
        """Create a parameterised subclass with category keys from *filter_schema*.

        *filter_schema* must be an ``AgentState`` subclass.  Category keys
        are the field names that *filter_schema* adds beyond ``AgentState``.
        """
        base_keys = frozenset(AgentState.__annotations__.keys())
        return type(
            f"{cls.__name__}[{filter_schema.__name__}]",
            (cls,),
            {
                "state_schema": filter_schema,
                "_category_keys": (
                    frozenset(filter_schema.__annotations__.keys()) - base_keys
                ),
                "tools": [],
            },
        )

    # ── Classification helpers ──────────────────────────────────────────

    def _get_classification(self, state: dict[str, Any]) -> dict[str, str]:
        """Extract classification values from flat state keys."""
        return {k: state[k] for k in self._category_keys if state.get(k)}

    # ── Skill filtering ─────────────────────────────────────────────────

    def _filter_skills(
        self,
        all_skills: list[SkillMetadata],
        classification: dict[str, str],
    ) -> list[SkillMetadata]:
        """Keep skills whose **all** category values match *classification*.

        Universal skills (no category metadata) always match.  Skills with
        metadata keys absent from *classification* are excluded (the
        ``classification.get(k)`` returns ``None``, failing the equality
        check).

        Results are sorted by specificity (fewest categories first) so the
        agent reads from general to specific.
        """
        category_keys = self._category_keys
        matched: list[SkillMetadata] = []
        for skill in all_skills:
            meta = skill.get("metadata") or {}
            categories = {k: v for k, v in meta.items() if k in category_keys}
            if all(classification.get(k) == v for k, v in categories.items()):
                matched.append(skill)

        matched.sort(
            key=lambda s: len(
                {
                    k: v
                    for k, v in (s.get("metadata") or {}).items()
                    if k in category_keys
                }
            )
        )
        return matched

    # ── Context formatting ──────────────────────────────────────────────

    def _format_context(self, classification: dict[str, str]) -> str:
        """Build a system prompt section describing the active classification."""
        lines = [
            f"## {self._context_header}\n",
            self._context_intro,
        ]
        for k, v in classification.items():
            lines.append(f"- **{k}**: {v}")
        lines.append("")
        lines.append(self._context_outro)
        return "\n".join(lines)

    # ── Middleware hooks ─────────────────────────────────────────────────

    async def abefore_agent(
        self,
        state: dict[str, Any],
        runtime: Any,
    ) -> dict[str, Any] | None:
        """Filter skills_metadata by classification from flat state keys."""
        classification = self._get_classification(state)
        skills = state.get("skills_metadata")
        if not classification or not skills:
            return None
        return {"skills_metadata": self._filter_skills(skills, classification)}

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """Inject classification context into the system prompt."""
        classification = self._get_classification(request.state)
        if classification:
            context = self._format_context(classification)
            new_sys = append_to_system_message(request.system_message, context)
            return await handler(request.override(system_message=new_sys))
        return await handler(request)
