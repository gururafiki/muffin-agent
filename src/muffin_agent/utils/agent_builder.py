"""Fluent builder for muffin agents (deep and ReAct).

``MuffinAgentBuilder`` reads top-to-bottom like a recipe.  Every capability
is an explicit ``with_*`` call that registers the required backend routes,
middleware, tools, and system-prompt partials in lock-step.  Two terminal
methods emit the agent:

* :meth:`MuffinAgentBuilder.build_deep_agent` forwards to
  :func:`deepagents.create_deep_agent`.
* :meth:`MuffinAgentBuilder.build_react_agent` forwards to
  :func:`langchain.agents.create_agent`.

Deep-agent-only capabilities (``with_subagents``, ``with_skills``,
``with_permission``) raise ``ValueError`` if the caller ends with
``build_react_agent``.

Universal middleware defaults — added automatically by every ``build_*``
call before caller-supplied middleware:

1. :class:`langchain.agents.middleware.ModelCallLimitMiddleware` —
   *conditional*. Added only when :meth:`with_model_call_limit` was called.
   Caps total LLM calls per run/thread before any other middleware runs.
2. :class:`langchain.agents.middleware.ToolCallLimitMiddleware` —
   *conditional, may repeat*. Added once per :meth:`with_tool_call_limit`
   call so per-tool and global caps can co-exist.
3. :class:`langchain.agents.middleware.ModelFallbackMiddleware` —
   *conditional*. Added only when :meth:`with_fallback_models` was called.
   Tries the primary model; on exception, walks the fallback chain.
4. :class:`langchain.agents.middleware.ModelRetryMiddleware` — retries
   transient and mid-stream LLM provider errors that escape the SDK's
   connect-time retry budget.  Filtered via :func:`_should_retry_llm_call`
   to skip permanent (auth/perm/validation) errors.
5. :class:`langchain.agents.middleware.ContextEditingMiddleware` —
   *conditional*. Added only when :meth:`with_context_editing` was called.
   Trims older tool outputs when the message-window token count exceeds
   the trigger; cheap pre-summarisation cleanup.
6. :class:`langchain.agents.middleware.SummarizationMiddleware` —
   *conditional*. Added only when :meth:`with_summarization` was called.
   Summarises older messages with an LLM when context-editing alone
   cannot cap the window.
7. :class:`ToolKnowledgeMiddleware`
8. :class:`ToolResultCacheMiddleware`
9. :class:`langchain.agents.middleware.ToolRetryMiddleware` — retries
   transient tool errors (HTTP 5xx, gateway, network) via
   :func:`_should_retry_tool_call`.
10. *(ReAct only)* :class:`FilesystemMiddleware` — wires the composite
    backend into the filesystem tools.  Deep agents receive the composite
    through the ``backend=`` kwarg instead.
11. *(ReAct only, conditional)* :class:`MemoryMiddleware` — added when
    :meth:`with_persistent_memory` was called.  Deep agents get the same
    middleware implicitly via ``create_deep_agent(memory=...)``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Self

import anthropic
import openai
from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, FilesystemBackend
from deepagents.backends.protocol import BackendFactory, BackendProtocol
from deepagents.backends.state import StateBackend
from deepagents.backends.store import StoreBackend
from deepagents.middleware.filesystem import FilesystemMiddleware
from deepagents.middleware.memory import MemoryMiddleware
from deepagents.middleware.permissions import FilesystemPermission
from deepagents.middleware.subagents import CompiledSubAgent, SubAgent
from langchain.agents import create_agent
from langchain.agents.middleware import (
    ClearToolUsesEdit,
    ContextEditingMiddleware,
    ModelCallLimitMiddleware,
    ModelFallbackMiddleware,
    ModelRetryMiddleware,
    SummarizationMiddleware,
    ToolCallLimitMiddleware,
    ToolRetryMiddleware,
)
from langchain.agents.middleware.summarization import ContextSize
from langchain.agents.middleware.types import AgentMiddleware
from langchain.agents.structured_output import AutoStrategy, ResponseFormat
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage
from langchain_core.tools import BaseTool, ToolException
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.base import BaseStore
from langgraph.types import Checkpointer

from ..middlewares import (
    CollectionFindings,
    SubagentRefinementMiddleware,
    SubagentRefinementParentMiddleware,
    ToolKnowledgeMiddleware,
    ToolResultCacheMiddleware,
)
from ..prompts import render_template
from ..sandbox import get_backend
from .backends import _SKILLS_ROOT, _memories_namespace

# Permanent provider errors — never retry. Checked first because most are
# subclasses of ``APIError`` and would otherwise be caught by the transient
# tuple below.
_PERMANENT_LLM_ERRORS: tuple[type[BaseException], ...] = (
    openai.AuthenticationError,
    openai.PermissionDeniedError,
    openai.BadRequestError,
    anthropic.AuthenticationError,
    anthropic.PermissionDeniedError,
    anthropic.BadRequestError,
)

# Transient provider errors — retry. ``APIConnectionError`` covers
# ``APITimeoutError``. Bare ``APIError`` is included to catch mid-stream
# "Provider returned error" cases raised directly by the SDK's stream
# iterator (the SDK's own ``max_retries`` does not cover these).
_TRANSIENT_LLM_ERRORS: tuple[type[BaseException], ...] = (
    openai.APIConnectionError,
    openai.RateLimitError,
    openai.InternalServerError,
    openai.APIError,
    anthropic.APIConnectionError,
    anthropic.RateLimitError,
    anthropic.InternalServerError,
    anthropic.APIError,
)


def _should_retry_llm_call(exc: Exception) -> bool:
    """Filter for ``ModelRetryMiddleware``: only transient provider errors."""
    if isinstance(exc, _PERMANENT_LLM_ERRORS):
        return False
    return isinstance(exc, _TRANSIENT_LLM_ERRORS)


# Substrings in ``ToolException.args[0]`` that mark a transient tool failure
# worth retrying (server-side / network hiccups). Argument-validation and
# missing-credential errors do NOT appear here — they are permanent for the
# given call shape and re-firing them just burns budget.
_TRANSIENT_TOOL_HINTS: tuple[str, ...] = (
    "http error 500",
    "http error 502",
    "http error 503",
    "http error 504",
    "bad gateway",
    "service unavailable",
    "gateway timeout",
    "internal server error",
    "connection error",
    "connection reset",
    "connection refused",
    "timeout",
    "timed out",
)


def _should_retry_tool_call(exc: Exception) -> bool:
    """Filter for ``ToolRetryMiddleware``: only transient tool errors.

    ``langchain-mcp-adapters`` wraps MCP-side HTTP failures in
    :class:`ToolException` whose string body includes the upstream HTTP
    status (e.g. ``"HTTP error 502: Bad Gateway"``). Match on canonical
    5xx codes plus obvious network errors. Anything else (4xx, validation,
    missing credential) propagates so the LLM can adapt.
    """
    if isinstance(exc, ToolException):
        message = str(exc).lower()
        return any(hint in message for hint in _TRANSIENT_TOOL_HINTS)
    return isinstance(exc, TimeoutError | ConnectionError)


# Prompt partials registered by each capability.  Rendered once and appended
# to the caller's system prompt in insertion order, with duplicates removed.
_PARTIAL_SANDBOX = "sandbox.jinja"
_PARTIAL_TOOL_RESULT_CACHE = "middlewares/tool_result_cache.jinja"
_PARTIAL_SHORT_TERM_MEMORY = "middlewares/short_term_memory.jinja"
_PARTIAL_PERSISTENT_MEMORY = "middlewares/persistent_memory.jinja"
_PARTIAL_SKILLS = "middlewares/skills.jinja"

ToolLike = BaseTool | Callable[..., Any] | dict[str, Any]
# AgentMiddleware is generic; the middleware stack holds middlewares with
# heterogeneous state/context/response types that all share the public
# ``AgentMiddleware`` protocol.  Using ``[Any, Any, Any]`` matches how
# ``create_deep_agent`` / ``create_agent`` declare their ``middleware=``
# kwargs (unparameterised ``Sequence[AgentMiddleware]``).
AnyMiddleware = AgentMiddleware[Any, Any, Any]


@dataclass
class _PromptConfig:
    """Base system prompt plus ordered, de-duplicated capability partials."""

    base: str | SystemMessage | None = None
    partials: list[str] = field(default_factory=list)

    def add_partial(self, name: str) -> None:
        if name not in self.partials:
            self.partials.append(name)

    def render(self) -> str | SystemMessage | None:
        if not self.partials and self.base is None:
            return None
        suffix = "\n\n".join(render_template(p) for p in self.partials)
        if isinstance(self.base, SystemMessage):
            existing = self.base.content if isinstance(self.base.content, str) else ""
            if existing and suffix:
                combined: str = f"{existing}\n\n{suffix}"
            else:
                combined = existing or suffix
            return SystemMessage(content=combined, id=self.base.id, name=self.base.name)
        base_str = self.base or ""
        if not suffix:
            return base_str or None
        return f"{base_str}\n\n{suffix}" if base_str else suffix


@dataclass
class _BackendConfig:
    """Composite-backend composition: default maker + pre-built routes."""

    # Only the sandbox default needs runtime at build time — other backends
    # read state/store/config via langgraph context helpers and are eagerly
    # instantiated on capability registration.
    default_maker: BackendFactory | None = None
    routes: dict[str, BackendProtocol] = field(default_factory=dict)
    memory_sources: list[str] = field(default_factory=list)

    def factory(self) -> BackendFactory | None:
        if self.default_maker is None and not self.routes:
            return None
        default_maker = self.default_maker
        routes = dict(self.routes)

        def make(runtime: Any) -> BackendProtocol:
            default: BackendProtocol = (
                default_maker(runtime) if default_maker is not None else StateBackend()
            )
            return CompositeBackend(default=default, routes=routes)

        return make


@dataclass
class _ToolConfig:
    """Accumulated tools plus per-tool cacheable flag."""

    tools: list[ToolLike] = field(default_factory=list)
    cacheable_names: set[str] = field(default_factory=set)

    @property
    def cacheable_frozen(self) -> frozenset[str] | None:
        # Match v1 semantics: cache-all when no tools registered.
        if not self.tools:
            return None
        return frozenset(self.cacheable_names)


@dataclass
class _SkillConfig:
    """Skill bundle — at most one per agent."""

    called: bool = False
    paths: list[str] = field(default_factory=list)
    filter_middleware: AnyMiddleware | None = None


class MuffinAgentBuilder:
    """Fluent builder for muffin deep agents and ReAct agents.

    Example:
        >>> agent = (
        ...     MuffinAgentBuilder(llm, name="market_regime")
        ...     .with_system_prompt_template("investment/market_regime.jinja")
        ...     .with_sandbox()
        ...     .with_short_term_memory()
        ...     .with_persistent_memory()
        ...     .with_subagents(subagents)
        ...     .with_tool(compute_yield_curve_metrics)
        ...     .with_response_format(AutoStrategy(schema=MarketRegimeOutput))
        ...     .with_store(store)
        ...     .build_deep_agent()
        ... )
    """

    def __init__(
        self,
        model: str | BaseChatModel,
        *,
        name: str | None = None,
    ) -> None:
        """Initialise the builder.

        Args:
            model: LangChain chat model instance or provider string.  Required
                by both ``create_agent`` (langchain) and (de-facto) by
                ``create_deep_agent``.
            name: Optional agent name forwarded to the underlying factory.
                Useful for LangSmith traces.
        """
        self._model: str | BaseChatModel = model
        self._name: str | None = name
        self._prompt = _PromptConfig()
        self._backend = _BackendConfig()
        self._tools = _ToolConfig()
        self._skills = _SkillConfig()
        self._subagents: list[SubAgent | CompiledSubAgent] = []
        self._middleware: list[AnyMiddleware] = []
        self._fallback_models: list[str | BaseChatModel] = []
        self._context_editing: ContextEditingMiddleware | None = None
        self._summarization: SummarizationMiddleware | None = None
        self._model_call_limit: ModelCallLimitMiddleware | None = None
        self._tool_call_limits: list[ToolCallLimitMiddleware] = []
        self._knowledge_summariser: BaseChatModel | None = None
        self._subagent_refinement: bool = False
        self._permissions: list[FilesystemPermission] = []
        self._response_format: ResponseFormat[Any] | type | dict[str, Any] | None = None
        self._store: BaseStore | None = None
        self._context_schema: type | None = None
        self._checkpointer: Checkpointer | None = None

    # ─── System prompt ───────────────────────────────────────────────────

    def with_system_prompt(self, prompt: str | SystemMessage) -> Self:
        """Set the base system prompt from a string or ``SystemMessage``.

        Mutually exclusive with :meth:`with_system_prompt_template` — the
        last call wins.  Both underlying factories accept the same union.
        """
        self._prompt.base = prompt
        return self

    def with_system_prompt_template(self, template: str, **vars: object) -> Self:
        """Render a Jinja template and use the result as the base prompt.

        Hides the direct :func:`render_template` call.  Mutually exclusive
        with :meth:`with_system_prompt` — the last call wins.
        """
        self._prompt.base = render_template(template, **vars)
        return self

    # ─── Backend capabilities ────────────────────────────────────────────

    def with_sandbox(self) -> Self:
        """Route the default (prefix-less) path to the OpenSandbox backend.

        Required if the agent uses ``execute_python`` or writes to
        un-prefixed sandbox paths like ``/data/cache/...``.
        """
        self._backend.default_maker = get_backend
        self._prompt.add_partial(_PARTIAL_SANDBOX)
        return self

    def with_short_term_memory(self, path: str = "/scratch/") -> Self:
        """Mount thread-scoped memory at *path* backed by graph state.

        Survives sandbox recycling within the current thread; lost when
        the thread ends.
        """
        self._backend.routes[path] = StateBackend()
        self._prompt.add_partial(_PARTIAL_SHORT_TERM_MEMORY)
        return self

    def with_persistent_memory(
        self,
        path: str = "/memories/",
        filename: str = "AGENTS.md",
    ) -> Self:
        """Mount per-user durable memory at *path* backed by the store.

        Deep agents receive :class:`MemoryMiddleware` implicitly via
        ``create_deep_agent(memory=[...])``; ReAct agents get the same
        middleware wired explicitly in :meth:`_assemble_middleware`.
        Both load ``{path}{filename}`` into the system prompt on every
        turn.  The ``/memories/`` namespace is resolved per-request from
        ``configurable.user_id`` (with :envvar:`MEMORY_DEBUG_USER_ID` as
        a debug fallback — see :class:`MemoryConfiguration`).
        """
        self._backend.routes[path] = StoreBackend(namespace=_memories_namespace)
        source = f"{path.rstrip('/')}/{filename}"
        if source not in self._backend.memory_sources:
            self._backend.memory_sources.append(source)
        self._prompt.add_partial(_PARTIAL_PERSISTENT_MEMORY)
        return self

    def with_skills(
        self,
        skills: list[str],
        *,
        skills_root: Path | str | None = None,
        filter_middleware: AnyMiddleware | None = None,
    ) -> Self:
        """Attach a bundle of skill directories (deep-agent only).

        Mounts ``/skills/`` on a :class:`FilesystemBackend` rooted at
        *skills_root* (defaults to the muffin skills package) and forwards
        the provided paths to ``create_deep_agent(skills=...)``.

        ``filter_middleware`` is appended to the middleware stack when
        provided.  ``SkillFilterMiddleware`` operates on the full
        ``skills_metadata`` state — it cannot be scoped to a specific
        directory today, so call this method at most once per agent.

        Raises ``ValueError`` if called twice.
        """
        if self._skills.called:
            raise ValueError(
                "with_skills(...) may be called at most once per agent. "
                "Per-directory filter scoping requires extending "
                "SkillFilterMiddleware — see the deferred items in the plan."
            )
        self._skills.called = True
        self._skills.paths = list(skills)
        self._skills.filter_middleware = filter_middleware
        root = Path(skills_root) if skills_root is not None else _SKILLS_ROOT
        self._backend.routes["/skills/"] = FilesystemBackend(
            root_dir=root, virtual_mode=True
        )
        self._prompt.add_partial(_PARTIAL_SKILLS)
        return self

    # ─── Tools and subagents ─────────────────────────────────────────────

    def with_subagents(self, subagents: Sequence[SubAgent | CompiledSubAgent]) -> Self:
        """Register subagents (deep-agent only).

        Raises at build time if followed by :meth:`build_react_agent`.
        """
        self._subagents.extend(subagents)
        return self

    def with_subagent_refinement(self) -> Self:
        """Enable the structured-findings + scratch-cache refinement protocol.

        Role is decided at build time from whether subagents are wired:

        * **Child** (no subagents wired) — registers
          :class:`SubagentRefinementMiddleware`, which forces a
          :class:`CollectionFindings` response (via ``response_format``),
          reads prior findings from
          ``/scratch/subagent_runs/<call_id>.json`` when the task
          description carries ``prior_call_id=<id>``, persists this run's
          findings on completion, and amends the system prompt with both
          the static refinement rules and (if present) the per-call
          prior-findings block.
        * **Parent** (subagents wired) — registers
          :class:`SubagentRefinementParentMiddleware`, which only amends
          the system prompt with the orchestrator's gap-handling rules
          so it knows how to read ``CollectionFindings.gaps`` and
          re-issue refinement calls with ``prior_call_id=<id>``.

        Idempotent — calling twice has no extra effect.
        """
        self._subagent_refinement = True
        return self

    def with_tool(
        self,
        tool: ToolLike,
        *,
        is_cacheable: bool = True,
        run_limit: int | None = None,
        thread_limit: int | None = None,
        exit_behavior: Literal["continue", "end", "error"] = "continue",
    ) -> Self:
        """Register a single tool, optionally with per-tool policy.

        Tools are cacheable by default — their name is added to the
        :class:`ToolResultCacheMiddleware` cacheable set.  Pass
        ``is_cacheable=False`` to keep the tool but exclude it from
        caching.

        Pass ``run_limit`` and/or ``thread_limit`` to cap how many times
        this specific tool may be called per run / per thread.  When a
        cap is set the builder appends a dedicated
        :class:`ToolCallLimitMiddleware` scoped to this tool's name.
        Use :meth:`with_tool_call_limit` (no ``tool_name``) for global
        caps that span all tools.

        Limits require the tool to expose a string ``.name`` attribute
        (i.e. a real :class:`BaseTool`); raw provider-tool dicts cannot
        be limited per-tool — apply a global cap instead.
        """
        self._tools.tools.append(tool)
        name = getattr(tool, "name", None)
        if is_cacheable and isinstance(name, str):
            self._tools.cacheable_names.add(name)
        if run_limit is not None or thread_limit is not None:
            if not isinstance(name, str):
                raise ValueError(
                    "Per-tool call limits require a tool with a `.name` attribute. "
                    "Use with_tool_call_limit(...) for a global cap instead."
                )
            self._tool_call_limits.append(
                ToolCallLimitMiddleware(
                    tool_name=name,
                    run_limit=run_limit,
                    thread_limit=thread_limit,
                    exit_behavior=exit_behavior,
                )
            )
        self._prompt.add_partial(_PARTIAL_TOOL_RESULT_CACHE)
        return self

    # ─── Permissions and extra middleware ────────────────────────────────

    def with_permission(self, permission: FilesystemPermission) -> Self:
        """Append a single :class:`FilesystemPermission` rule (deep-agent only).

        Forwarded to ``create_deep_agent(permissions=...)``.  Rules are
        evaluated in declaration order; the first match wins.
        """
        self._permissions.append(permission)
        return self

    def with_middleware(self, middleware: AnyMiddleware) -> Self:
        """Append a single middleware instance after universal defaults."""
        self._middleware.append(middleware)
        return self

    def with_fallback_models(self, *models: str | BaseChatModel) -> Self:
        """Register fallback models tried in order when the primary errors.

        Wires :class:`ModelFallbackMiddleware` automatically. Each model
        receives the same retry budget as the primary (via the universal
        :class:`ModelRetryMiddleware`); fallback only kicks in after retries
        on the current model are exhausted.

        Pair with :meth:`ModelConfiguration.get_llm_for_role`:
            chain = config.get_llm_for_role("orchestrator")
            builder = MuffinAgentBuilder(chain[0]).with_fallback_models(*chain[1:])
        """
        self._fallback_models.extend(models)
        return self

    def with_context_editing(
        self,
        *,
        trigger: int = 40_000,
        keep: int = 4,
        clear_tool_inputs: bool = False,
    ) -> Self:
        """Wire :class:`ContextEditingMiddleware` with a ``ClearToolUsesEdit``.

        When the message-window token count exceeds *trigger*, older tool
        outputs are replaced with a ``[cleared]`` placeholder while the
        most recent *keep* tool messages remain verbatim. Reduces context
        bloat for tool-heavy agents (e.g. data-collection ReAct loops)
        without an LLM call.

        Last call wins; calling twice replaces the prior configuration.
        """
        self._context_editing = ContextEditingMiddleware(
            edits=[
                ClearToolUsesEdit(
                    trigger=trigger,
                    keep=keep,
                    clear_tool_inputs=clear_tool_inputs,
                )
            ]
        )
        return self

    def with_summarization(
        self,
        model: str | BaseChatModel | None = None,
        *,
        trigger: ContextSize | list[ContextSize] = ("tokens", 80_000),
        keep: ContextSize = ("messages", 20),
    ) -> Self:
        """Wire upstream :class:`SummarizationMiddleware` as a fallback.

        Summarises older messages with *model* (defaults to the agent's
        primary model) when *trigger* is exceeded; preserves *keep* of
        recent context. Use as a fallback after :meth:`with_context_editing`
        — context-editing is cheaper (no LLM call) and runs first.

        Deep agents already include their own summarisation middleware via
        ``create_deep_agent``; calling this on a deep agent stacks them and
        is rarely useful.

        Last call wins.
        """
        summarisation_model = model if model is not None else self._model
        self._summarization = SummarizationMiddleware(
            model=summarisation_model,
            trigger=trigger,
            keep=keep,
        )
        return self

    def with_model_call_limit(
        self,
        *,
        run_limit: int | None = None,
        thread_limit: int | None = None,
        exit_behavior: Literal["end", "error"] = "end",
    ) -> Self:
        """Cap the number of model calls per run and/or per thread.

        Wires upstream :class:`ModelCallLimitMiddleware`. When the cap is
        hit with ``exit_behavior="end"`` the agent stops gracefully with
        an injected ``AIMessage`` describing the limit; with ``"error"``
        it raises ``ModelCallLimitExceededError``.

        Last call wins.
        """
        self._model_call_limit = ModelCallLimitMiddleware(
            run_limit=run_limit,
            thread_limit=thread_limit,
            exit_behavior=exit_behavior,
        )
        return self

    def with_tool_knowledge(
        self,
        summariser: BaseChatModel,
    ) -> Self:
        """Use *summariser* to generate one-line lessons from tool failures.

        :class:`ToolKnowledgeMiddleware` is universal — it always runs.
        By default lessons are recorded as a deterministic ``tool: error``
        fallback string. Pass a small/cheap chat model here (e.g. Haiku
        4.5) to upgrade fallback strings to LLM-summarised, action-
        oriented hints. The summariser is called at most once per unique
        ``(tool, error_class)`` pair per session — repeat errors hit the
        store cache and pay zero LLM cost.
        """
        self._knowledge_summariser = summariser
        return self

    def with_tool_call_limit(
        self,
        *,
        tool_name: str | None = None,
        run_limit: int | None = None,
        thread_limit: int | None = None,
        exit_behavior: Literal["continue", "end", "error"] = "continue",
    ) -> Self:
        """Cap tool calls globally or for a specific tool name.

        Wires upstream :class:`ToolCallLimitMiddleware`. With no
        ``tool_name`` the cap spans every tool in the agent; with a name
        the cap is per-tool. Multiple calls accumulate so global and
        per-tool caps can co-exist.

        For per-tool caps on tools you're registering yourself, prefer
        the ``run_limit`` / ``thread_limit`` kwargs on :meth:`with_tool`
        — they keep the policy declared at the same site as the tool.
        Reach for this method when capping all tools at once or capping
        a tool you don't register through :meth:`with_tool` (e.g. the
        deep-agent ``task`` tool injected by the framework).
        """
        self._tool_call_limits.append(
            ToolCallLimitMiddleware(
                tool_name=tool_name,
                run_limit=run_limit,
                thread_limit=thread_limit,
                exit_behavior=exit_behavior,
            )
        )
        return self

    # ─── Miscellaneous kwargs ────────────────────────────────────────────

    def with_response_format(
        self,
        response_format: ResponseFormat[Any] | type | dict[str, Any],
    ) -> Self:
        """Attach a structured-output strategy (e.g. ``AutoStrategy``)."""
        self._response_format = response_format
        return self

    def with_store(self, store: BaseStore) -> Self:
        """Attach a LangGraph :class:`BaseStore` instance."""
        self._store = store
        return self

    def with_context_schema(self, schema: type) -> Self:
        """Attach a context schema type."""
        self._context_schema = schema
        return self

    def with_checkpointer(self, checkpointer: Checkpointer) -> Self:
        """Attach a checkpointer for graph state persistence."""
        self._checkpointer = checkpointer
        return self

    # ─── Terminal builders ───────────────────────────────────────────────

    def build_deep_agent(self) -> CompiledStateGraph:
        """Emit a compiled deep agent via :func:`deepagents.create_deep_agent`.

        When :meth:`with_persistent_memory` was called, ``memory=`` is
        forwarded to ``create_deep_agent``, which installs the stock
        :class:`MemoryMiddleware` implicitly.
        """
        return create_deep_agent(
            model=self._model,
            tools=self._tools.tools or None,
            system_prompt=self._prompt.render(),
            middleware=self._assemble_middleware(is_deep=True, backend_factory=None),
            subagents=self._subagents or None,
            skills=self._skills.paths or None,
            permissions=self._permissions or None,
            response_format=self._effective_response_format(),
            context_schema=self._context_schema,
            checkpointer=self._checkpointer,
            store=self._store,
            backend=self._backend.factory(),
            memory=self._backend.memory_sources or None,
            name=self._name,
        )

    def build_react_agent(self) -> CompiledStateGraph:
        """Emit a compiled ReAct agent via :func:`langchain.agents.create_agent`."""
        if self._skills.called:
            raise ValueError(
                "with_skills(...) is only supported with build_deep_agent(). "
                "Use build_deep_agent() or drop the skills bundle."
            )
        if self._subagents:
            raise ValueError(
                "with_subagents(...) is only supported with build_deep_agent(). "
                "Use build_deep_agent() or drop the subagents."
            )
        if self._permissions:
            raise ValueError(
                "with_permission(...) is only supported with build_deep_agent(). "
                "Use build_deep_agent() or drop the permissions."
            )
        backend_factory = self._backend.factory()
        return create_agent(
            model=self._model,
            tools=self._tools.tools or None,
            system_prompt=self._prompt.render(),
            middleware=self._assemble_middleware(
                is_deep=False, backend_factory=backend_factory
            ),
            response_format=self._effective_response_format(),
            context_schema=self._context_schema,
            checkpointer=self._checkpointer,
            store=self._store,
            name=self._name,
        )

    def _effective_response_format(
        self,
    ) -> ResponseFormat[Any] | type | dict[str, Any] | None:
        """Return ``response_format`` honouring the refinement contract.

        On a child subagent (no subagents wired) with refinement enabled,
        default to ``AutoStrategy(CollectionFindings)`` so the runtime
        validates the structured response. Caller-set ``response_format``
        always wins.
        """
        if self._response_format is not None:
            return self._response_format
        if self._subagent_refinement and not self._subagents:
            return AutoStrategy(schema=CollectionFindings)
        return None

    # ─── Internals ───────────────────────────────────────────────────────

    def _assemble_middleware(
        self,
        *,
        is_deep: bool,
        backend_factory: BackendFactory | None,
    ) -> list[AnyMiddleware]:
        """Assemble the final middleware stack.

        Order:
        1. :class:`ModelCallLimitMiddleware` (outermost, conditional) —
           registered only when :meth:`with_model_call_limit` was called.
           Caps total LLM calls per run/thread; on cap, jumps to end
           with an injected ``AIMessage`` (or raises) before any other
           middleware runs.
        2. :class:`ToolCallLimitMiddleware` (conditional, may repeat) —
           registered once per :meth:`with_tool_call_limit` call.
           Multiple instances co-exist so per-tool and global caps can
           be combined.
        3. :class:`ModelFallbackMiddleware` (conditional) — tries the
           primary model; on exception, walks the caller-supplied
           fallback chain.  Registered only when
           :meth:`with_fallback_models` was called.
        4. :class:`ModelRetryMiddleware` (LangChain upstream, universal) —
           retries transient and mid-stream LLM provider errors that
           escape the SDK's connect-time retry budget.  Filters via
           :func:`_should_retry_llm_call` to skip permanent errors
           (auth/perm/validation).  ``on_failure="error"`` propagates
           the exception after exhaustion so the outer fallback can
           switch models.
        5. :class:`ContextEditingMiddleware` (conditional) — registered
           only when :meth:`with_context_editing` was called.  Runs
           inside the fallback/retry boundary so the same edited message
           list is sent on every attempt.
        6. :class:`SummarizationMiddleware` (conditional) — registered
           only when :meth:`with_summarization` was called.  Sits
           inside context-editing so cheap edits run first; the LLM
           summarisation only fires when edits cannot cap the window.
        7. :class:`ToolKnowledgeMiddleware` (universal) — catches tool
           errors, blocks duplicates, and learns one-line lessons
           (LLM-summarised when a summariser is configured via
           :meth:`with_tool_knowledge`).
        8. :class:`ToolResultCacheMiddleware` (universal)
        9. :class:`langchain.agents.middleware.ToolRetryMiddleware`
           (universal) — retries transient tool errors (HTTP 5xx /
           gateway timeouts / network) via :func:`_should_retry_tool_call`.
           Sits below the cache so cache hits short-circuit and don't
           consume retry budget.
        10. *(ReAct only)* :class:`FilesystemMiddleware` when a backend is
            configured.
        11. *(ReAct only, conditional)* :class:`MemoryMiddleware` when
            :meth:`with_persistent_memory` was called.  Deep agents get
            the same middleware implicitly via
            ``create_deep_agent(memory=...)``.
        12. *(conditional)* Subagent-refinement middleware when
            :meth:`with_subagent_refinement` was called.  Picks the
            class based on role:
            - :class:`SubagentRefinementParentMiddleware` when subagents
              are wired (only amends the system prompt with rules).
            - :class:`SubagentRefinementMiddleware` otherwise (full child
              behaviour: reads/writes ``/scratch/subagent_runs/<call_id>.json``
              and injects prior findings into the system prompt).
        13. Optional skill-filter middleware registered via
            :meth:`with_skills`.
        14. Caller-supplied middleware (via :meth:`with_middleware`).
        """
        stack: list[AnyMiddleware] = []
        if self._model_call_limit is not None:
            stack.append(self._model_call_limit)
        stack.extend(self._tool_call_limits)
        if self._fallback_models:
            stack.append(ModelFallbackMiddleware(*self._fallback_models))
        stack.append(
            ModelRetryMiddleware(
                max_retries=3,
                retry_on=_should_retry_llm_call,
                on_failure="error",
                backoff_factor=2.0,
                initial_delay=1.0,
                max_delay=30.0,
                jitter=True,
            )
        )
        if self._context_editing is not None:
            stack.append(self._context_editing)
        if self._summarization is not None:
            stack.append(self._summarization)
        stack.extend(
            [
                ToolKnowledgeMiddleware(summariser=self._knowledge_summariser),
                ToolResultCacheMiddleware(cacheable_tools=self._tools.cacheable_frozen),
                ToolRetryMiddleware(
                    max_retries=1,
                    retry_on=_should_retry_tool_call,
                    on_failure="continue",
                    initial_delay=1.0,
                    max_delay=10.0,
                    backoff_factor=2.0,
                    jitter=True,
                ),
            ]
        )
        if not is_deep and backend_factory is not None:
            stack.append(FilesystemMiddleware(backend=backend_factory))
        if not is_deep and self._backend.memory_sources:
            assert backend_factory is not None
            stack.append(
                MemoryMiddleware(
                    backend=backend_factory,
                    sources=list(self._backend.memory_sources),
                )
            )
        if self._subagent_refinement:
            if self._subagents:
                # Parent role — only amends the system prompt with rules.
                stack.append(SubagentRefinementParentMiddleware())
            else:
                # Child role — needs the backend to read/write
                # /scratch/subagent_runs/<call_id>.json. Deep agents
                # resolve the backend via create_deep_agent's own
                # factory; ReAct agents use the local backend_factory.
                refinement_factory = backend_factory or self._backend.factory()
                if refinement_factory is not None:
                    stack.append(
                        SubagentRefinementMiddleware(
                            backend_factory=refinement_factory,
                        )
                    )
        if self._skills.filter_middleware is not None:
            stack.append(self._skills.filter_middleware)
        stack.extend(self._middleware)
        return stack
