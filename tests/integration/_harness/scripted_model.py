"""The core LLM seam — a real ``BaseChatModel`` driven by a scripted timeline.

Why not the existing ``FakeLLM``? ``langchain.agents.create_agent`` (and deep
agents, which delegate to it) call ``model.bind_tools(...)``, and the base
``BaseChatModel.bind_tools`` raises ``NotImplementedError``. The repo's duck-typed
``FakeLLM`` therefore cannot drive a *real* ReAct loop. ``ScriptedChatModel`` is a
genuine ``BaseChatModel`` subclass, so the real builder, the real middleware
stack, the real tool-calling loop, and the real structured-output composition all
run unchanged — only the bytes the "LLM" emits are scripted.

Authoring model: build an ordered list of *turns* and feed it to
:func:`patch_llm`. The N-th model call anywhere in the graph returns ``script[N]``:

* :func:`tool_turn` — an ``AIMessage`` carrying a tool call. Use it both for a
  real MCP/sandbox tool invocation AND for the final ``response_format`` turn
  (whose tool name is the response schema's class name, e.g. ``PeterLynchRawData``).
* :func:`final` — a free-form ``AIMessage`` (a plain text answer).
* a bare Pydantic ``BaseModel`` instance — consumed by the direct-call
  ``with_structured_output`` path (``ModelConfiguration.get_chat_model_for_role``).
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator, Sequence
from unittest.mock import MagicMock, patch

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel, ConfigDict

# A scripted turn is one of: a ready AIMessage, a string (wrapped in AIMessage),
# or a Pydantic BaseModel instance (returned by the structured-output path).
ScriptItem = Any


class ScriptExhaustedError(AssertionError):
    """Raised when the graph requests more model calls than the script provides.

    Surfaces a script/agent mismatch loudly instead of silently looping —
    the message reports how many turns were consumed so the author can extend
    the script to match the real call count.
    """


class Script:
    """An ordered, shared cursor over scripted model turns.

    A single ``Script`` instance is shared by every ``ScriptedChatModel`` built
    during one graph run, so the script is one timeline spanning all nodes.
    """

    def __init__(self, items: Sequence[ScriptItem]) -> None:
        self._items = list(items)
        self._i = 0
        #: One entry per model call — the input the graph sent (a list of
        #: ``BaseMessage`` for tool-loop turns, or the ``LanguageModelInput`` for
        #: the structured-output path). Use to assert on rendered prompts.
        self.inputs: list[Any] = []

    @property
    def consumed(self) -> int:
        """Number of turns consumed so far."""
        return self._i

    def advance(self, call_input: Any = None) -> ScriptItem:
        """Return the next scripted turn, or raise if the script is exhausted."""
        self.inputs.append(call_input)
        if self._i >= len(self._items):
            raise ScriptExhaustedError(
                f"ScriptedChatModel ran out of turns after {self._i} model call(s) "
                f"(script length {len(self._items)}). The graph made more LLM calls "
                "than scripted — extend the script to match, or check for an "
                "unexpected loop."
            )
        item = self._items[self._i]
        self._i += 1
        return item

    def last_system_prompt(self) -> str:
        """System-message content of the most recent model call (best effort)."""
        return self._last_content_of_type("system")

    def last_human_prompt(self) -> str:
        """First human-message content of the most recent model call.

        The task/input is seeded here by ``_InputPromptMiddleware`` (never baked
        into the system prompt), so node-agent tests assert on this.
        """
        return self._last_content_of_type("human")

    def _last_content_of_type(self, msg_type: str) -> str:
        if not self.inputs:
            return ""
        call_input = self.inputs[-1]
        if isinstance(call_input, (list, tuple)):
            for msg in call_input:
                if getattr(msg, "type", None) == msg_type:
                    return str(getattr(msg, "content", ""))
        return ""


class ScriptedChatModel(BaseChatModel):
    """A real ``BaseChatModel`` that replays :class:`Script` turns."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    script: Script

    @property
    def _llm_type(self) -> str:
        return "scripted"

    # ReAct loop binds tools onto the model; record + return self so the same
    # scripted instance keeps serving turns.
    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> "ScriptedChatModel":
        object.__setattr__(self, "_bound_tools", list(tools))
        return self

    def _next_message(self, messages: Sequence[BaseMessage]) -> AIMessage:
        item = self.script.advance(list(messages))
        if isinstance(item, AIMessage):
            return item
        if isinstance(item, BaseMessage):  # be forgiving
            return AIMessage(content=item.content)
        if isinstance(item, str):
            return AIMessage(content=item)
        raise ScriptExhaustedError(
            "ScriptedChatModel._generate got a non-message script turn "
            f"({type(item).__name__}); a bare Pydantic turn is only valid on the "
            "with_structured_output (direct-call) path. Re-order the script."
        )

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:  # noqa: ANN001
        msg = self._next_message(messages)
        return ChatResult(generations=[ChatGeneration(message=msg)])

    async def _agenerate(
        self, messages, stop=None, run_manager=None, **kwargs
    ) -> ChatResult:  # noqa: ANN001
        msg = self._next_message(messages)
        return ChatResult(generations=[ChatGeneration(message=msg)])

    def with_structured_output(self, schema, **kwargs):  # noqa: ANN001, ARG002
        """Direct-call structured path (``get_chat_model_for_role(schema=...)``).

        Returns a real ``Runnable`` (so ``with_fallbacks`` / ``with_retry``
        compose for real) that pops the next script turn — expected to be the
        Pydantic instance the node should receive.
        """
        script = self.script

        def _pop(call_input: Any) -> Any:
            return script.advance(call_input)

        async def _apop(call_input: Any) -> Any:
            return script.advance(call_input)

        return RunnableLambda(_pop, afunc=_apop)


# ── Authoring helpers ─────────────────────────────────────────────────────────


def tool_turn(
    name: str, args: dict[str, Any], *, id: str | None = None, content: str = ""
) -> AIMessage:
    """An ``AIMessage`` that calls tool *name* with *args*.

    Use for a real MCP/sandbox tool call, or for the final ``response_format``
    turn (where *name* is the response schema's class name).
    """
    return AIMessage(
        content=content,
        tool_calls=[
            {
                "name": name,
                "args": args,
                "id": id or f"call_{name}",
                "type": "tool_call",
            }
        ],
    )


def parallel_tool_turn(*calls: tuple[str, dict[str, Any]]) -> AIMessage:
    """An ``AIMessage`` issuing several tool calls in one turn (parallel calls).

    Real tool-calling LLMs batch independent calls like this; the ``ToolNode``
    executes them all before the next model turn. Each *call* is a
    ``(tool_name, args)`` tuple.
    """
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": name,
                "args": args,
                "id": f"call_{name}_{i}",
                "type": "tool_call",
            }
            for i, (name, args) in enumerate(calls)
        ],
    )


def final(text: str) -> AIMessage:
    """A free-form text ``AIMessage`` (a plain answer, no tool calls)."""
    return AIMessage(content=text)


# ── The LLM seam ──────────────────────────────────────────────────────────────


@contextmanager
def patch_llm(*script: ScriptItem) -> Iterator[Script]:
    """Patch the single LLM chokepoint with a scripted timeline.

    Patches ``ModelConfiguration.from_runnable_config`` so BOTH construction
    surfaces resolve to ``ScriptedChatModel`` instances sharing one cursor:

    * the factory ReAct path (``get_llm_for_role`` → ``MuffinAgentBuilder``), and
    * the direct-node path (``get_chat_model_for_role`` classmethod, which calls
      ``cls.from_runnable_config(config).get_llm_for_role(role)`` internally —
      see ``model_config.py``).

    Yields the shared :class:`Script` so tests can assert on ``cursor.consumed``.
    """
    from muffin_agent.model_config import ModelConfiguration

    cursor = Script(script)

    def _factory(config: Any) -> MagicMock:  # noqa: ARG001 — config unused
        cfg = MagicMock(name="ScriptedModelConfiguration")
        cfg.get_llm_for_role.side_effect = lambda role, **kw: [
            ScriptedChatModel(script=cursor)
        ]
        cfg.get_llm.side_effect = lambda *a, **k: ScriptedChatModel(script=cursor)
        cfg.get_summariser.return_value = None
        return cfg

    with patch.object(ModelConfiguration, "from_runnable_config", side_effect=_factory):
        yield cursor


# ── Schema-routed model (for parallel, heterogeneous subagents) ────────────────


def _tool_name(tool: Any) -> str | None:
    return getattr(tool, "name", None) or (
        tool.get("name") if isinstance(tool, dict) else None
    )


class SchemaRoutedModel(BaseChatModel):
    """A real ``BaseChatModel`` that answers by **schema**, not call order.

    Parallel fan-out graphs (the council's 13 personas, the 4 trading analysts)
    interleave model calls non-deterministically, so an ordered ``Script`` can't
    work. This model is **stateless** — every call dispatches on what is asked:

    * ReAct ``response_format`` turn — when a bound tool's name is a key in
      *responses* (the response-schema's class name), emit a tool call for it with
      the registered args (``{}`` → schema defaults). Ends the ReAct loop.
    * Direct ``with_structured_output(schema)`` turn — return the registered
      instance for that schema (looked up by class, then by class name).

    Because there is no shared cursor, the same instance is safe across any number
    of concurrent subagents.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    responses: dict[Any, Any]

    @property
    def _llm_type(self) -> str:
        return "schema-routed"

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> "SchemaRoutedModel":
        object.__setattr__(self, "_bound_tools", list(tools))
        return self

    def _message(self) -> AIMessage:
        for tool in getattr(self, "_bound_tools", []):
            name = _tool_name(tool)
            if name in self.responses:
                args = self.responses[name]
                if isinstance(args, BaseModel):
                    args = args.model_dump()
                return AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": name,
                            "args": args,
                            "id": f"call_{name}",
                            "type": "tool_call",
                        }
                    ],
                )
        return AIMessage(content="(schema-routed: no structured tool bound)")

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:  # noqa: ANN001
        return ChatResult(generations=[ChatGeneration(message=self._message())])

    async def _agenerate(
        self, messages, stop=None, run_manager=None, **kwargs
    ) -> ChatResult:  # noqa: ANN001
        return ChatResult(generations=[ChatGeneration(message=self._message())])

    def with_structured_output(self, schema, **kwargs):  # noqa: ANN001, ARG002
        resp = self.responses.get(schema)
        if resp is None:
            resp = self.responses.get(getattr(schema, "__name__", None))

        def _pop(_input: Any) -> Any:
            return resp

        async def _apop(_input: Any) -> Any:
            return resp

        return RunnableLambda(_pop, afunc=_apop)


@contextmanager
def patch_llm_by_schema(responses: dict[Any, Any]) -> Iterator[dict[Any, Any]]:
    """Patch the LLM seam with a stateless :class:`SchemaRoutedModel`.

    *responses* mixes two key kinds:

    * ``"<ResponseSchemaName>": <args dict | BaseModel>`` — the ReAct
      ``response_format`` turn emits a tool call for that schema (``{}`` → defaults).
    * ``<SchemaClass>: <BaseModel instance>`` — the direct
      ``with_structured_output(<SchemaClass>)`` turn returns it.

    Safe under parallel fan-out (no shared cursor).
    """
    from muffin_agent.model_config import ModelConfiguration

    def _factory(config: Any) -> MagicMock:  # noqa: ARG001
        cfg = MagicMock(name="SchemaRoutedModelConfiguration")
        cfg.get_llm_for_role.side_effect = lambda role, **kw: [
            SchemaRoutedModel(responses=responses)
        ]
        cfg.get_llm.side_effect = lambda *a, **k: SchemaRoutedModel(responses=responses)
        cfg.get_summariser.return_value = None
        return cfg

    with patch.object(ModelConfiguration, "from_runnable_config", side_effect=_factory):
        yield responses
