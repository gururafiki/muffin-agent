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

1. :class:`ToolErrorHandlerMiddleware`
2. :class:`ToolResultCacheMiddleware`
3. *(ReAct only)* :class:`FilesystemMiddleware` — wires the composite
   backend into the filesystem tools.  Deep agents receive the composite
   through the ``backend=`` kwarg instead.
4. *(ReAct only, conditional)* :class:`MemoryMiddleware` — added when
   :meth:`with_persistent_memory` was called.  Deep agents get the same
   middleware implicitly via ``create_deep_agent(memory=...)``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Self

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
from langchain.agents.middleware.types import AgentMiddleware
from langchain.agents.structured_output import ResponseFormat
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage
from langchain_core.tools import BaseTool
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.base import BaseStore
from langgraph.types import Checkpointer

from ..middlewares import ToolErrorHandlerMiddleware, ToolResultCacheMiddleware
from ..prompts import render_template
from ..sandbox import get_backend
from .backends import _SKILLS_ROOT, _memories_namespace

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

    def with_tool(self, tool: ToolLike, *, is_cacheable: bool = True) -> Self:
        """Register a single tool.

        Tools are cacheable by default — their name is added to the
        :class:`ToolResultCacheMiddleware` cacheable set.  Pass
        ``is_cacheable=False`` to keep the tool but exclude it from
        caching.
        """
        self._tools.tools.append(tool)
        if is_cacheable:
            name = getattr(tool, "name", None)
            if isinstance(name, str):
                self._tools.cacheable_names.add(name)
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
            response_format=self._response_format,
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
            response_format=self._response_format,
            context_schema=self._context_schema,
            checkpointer=self._checkpointer,
            store=self._store,
            name=self._name,
        )

    # ─── Internals ───────────────────────────────────────────────────────

    def _assemble_middleware(
        self,
        *,
        is_deep: bool,
        backend_factory: BackendFactory | None,
    ) -> list[AnyMiddleware]:
        """Assemble the final middleware stack.

        Order:
        1. :class:`ToolErrorHandlerMiddleware` (universal)
        2. :class:`ToolResultCacheMiddleware` (universal)
        3. *(ReAct only)* :class:`FilesystemMiddleware` when a backend is
           configured.
        4. *(ReAct only, conditional)* :class:`MemoryMiddleware` when
           :meth:`with_persistent_memory` was called.  Deep agents get
           the same middleware implicitly via
           ``create_deep_agent(memory=...)``.
        5. Optional skill-filter middleware registered via
           :meth:`with_skills`.
        6. Caller-supplied middleware (via :meth:`with_middleware`).
        """
        stack: list[AnyMiddleware] = [
            ToolErrorHandlerMiddleware(),
            ToolResultCacheMiddleware(cacheable_tools=self._tools.cacheable_frozen),
        ]
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
        if self._skills.filter_middleware is not None:
            stack.append(self._skills.filter_middleware)
        stack.extend(self._middleware)
        return stack
