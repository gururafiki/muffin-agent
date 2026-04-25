"""Unit tests for ``muffin_agent.utils.agent_builder.MuffinAgentBuilder``."""

from unittest.mock import MagicMock, patch

import pytest
from deepagents.middleware.permissions import FilesystemPermission

_DEEP_PATCH = "muffin_agent.utils.agent_builder.create_deep_agent"
_REACT_PATCH = "muffin_agent.utils.agent_builder.create_agent"


def _deep_kwargs(mock_create_deep_agent):
    """Return the kwargs dict from the most recent ``create_deep_agent`` call."""
    assert mock_create_deep_agent.call_count == 1
    _, kwargs = mock_create_deep_agent.call_args
    return kwargs


def _react_kwargs(mock_create_agent):
    """Return the kwargs dict from the most recent ``create_agent`` call."""
    assert mock_create_agent.call_count == 1
    _, kwargs = mock_create_agent.call_args
    return kwargs


@pytest.mark.unit
class TestMinimalBuilders:
    def test_minimal_deep_agent_no_backend_universal_middleware(self):
        """Bare builder forwards model and universal middleware, no backend."""
        from muffin_agent.middlewares import (
            ToolErrorHandlerMiddleware,
            ToolResultCacheMiddleware,
        )
        from muffin_agent.utils.agent_builder import MuffinAgentBuilder

        model = MagicMock(name="llm")
        with patch(_DEEP_PATCH) as mock_cda:
            MuffinAgentBuilder(model).build_deep_agent()

        kwargs = _deep_kwargs(mock_cda)
        assert kwargs["model"] is model
        assert kwargs["backend"] is None
        assert kwargs["system_prompt"] is None
        mw = kwargs["middleware"]
        assert len(mw) == 2
        assert isinstance(mw[0], ToolErrorHandlerMiddleware)
        assert isinstance(mw[1], ToolResultCacheMiddleware)
        assert mw[1]._cacheable_tools is None

    def test_minimal_react_agent_no_filesystem_middleware(self):
        """No routes → no ``FilesystemMiddleware`` wired on a ReAct agent."""
        from deepagents.middleware.filesystem import FilesystemMiddleware

        from muffin_agent.utils.agent_builder import MuffinAgentBuilder

        with patch(_REACT_PATCH) as mock_ca:
            MuffinAgentBuilder(MagicMock()).build_react_agent()

        mw = _react_kwargs(mock_ca)["middleware"]
        assert not any(isinstance(m, FilesystemMiddleware) for m in mw)

    def test_name_forwarded(self):
        """``name`` constructor arg is forwarded to the factory."""
        from muffin_agent.utils.agent_builder import MuffinAgentBuilder

        with patch(_DEEP_PATCH) as mock_cda:
            MuffinAgentBuilder(MagicMock(), name="market_regime").build_deep_agent()

        assert _deep_kwargs(mock_cda)["name"] == "market_regime"


@pytest.mark.unit
class TestBackendComposition:
    def test_with_sandbox_sets_default_backend(self):
        """``with_sandbox`` routes the default path through ``get_backend``."""
        from deepagents.backends import CompositeBackend

        from muffin_agent.utils.agent_builder import MuffinAgentBuilder

        with (
            patch(_DEEP_PATCH) as mock_cda,
            patch("muffin_agent.utils.agent_builder.get_backend") as mock_sandbox,
        ):
            mock_sandbox.return_value = MagicMock(name="sandbox_backend")
            MuffinAgentBuilder(MagicMock()).with_sandbox().build_deep_agent()

        factory = _deep_kwargs(mock_cda)["backend"]
        composite = factory(MagicMock(name="runtime"))
        assert isinstance(composite, CompositeBackend)
        assert composite.default is mock_sandbox.return_value

    def test_with_short_term_memory_adds_route(self):
        """``with_short_term_memory`` mounts ``/scratch/`` on ``StateBackend``."""
        from deepagents.backends.state import StateBackend

        from muffin_agent.utils.agent_builder import MuffinAgentBuilder

        with patch(_DEEP_PATCH) as mock_cda:
            MuffinAgentBuilder(MagicMock()).with_short_term_memory().build_deep_agent()

        factory = _deep_kwargs(mock_cda)["backend"]
        composite = factory(MagicMock(name="runtime"))
        assert "/scratch/" in composite.routes
        assert isinstance(composite.routes["/scratch/"], StateBackend)

    def test_with_persistent_memory_deep_forwards_memory_kwarg(self):
        """Deep agent forwards ``memory=`` + wires the ``/memories/`` route."""
        from deepagents.backends.store import StoreBackend

        from muffin_agent.utils.agent_builder import MuffinAgentBuilder

        with patch(_DEEP_PATCH) as mock_cda:
            MuffinAgentBuilder(MagicMock()).with_persistent_memory().build_deep_agent()

        kwargs = _deep_kwargs(mock_cda)
        assert kwargs["memory"] == ["/memories/AGENTS.md"]
        composite = kwargs["backend"](MagicMock(name="runtime"))
        assert "/memories/" in composite.routes
        assert isinstance(composite.routes["/memories/"], StoreBackend)

    def test_with_persistent_memory_react_adds_memory_middleware(self):
        """ReAct agent gets ``MemoryMiddleware`` appended."""
        from deepagents.middleware.memory import MemoryMiddleware

        from muffin_agent.utils.agent_builder import MuffinAgentBuilder

        with patch(_REACT_PATCH) as mock_ca:
            MuffinAgentBuilder(MagicMock()).with_persistent_memory().build_react_agent()

        mw = _react_kwargs(mock_ca)["middleware"]
        assert any(isinstance(m, MemoryMiddleware) for m in mw)

    def test_backend_without_sandbox_falls_back_to_state_default(self):
        """Routes without sandbox default the prefix-less path to StateBackend."""
        from deepagents.backends.state import StateBackend

        from muffin_agent.utils.agent_builder import MuffinAgentBuilder

        with patch(_DEEP_PATCH) as mock_cda:
            MuffinAgentBuilder(MagicMock()).with_short_term_memory().build_deep_agent()

        factory = _deep_kwargs(mock_cda)["backend"]
        composite = factory(MagicMock(name="runtime"))
        assert isinstance(composite.default, StateBackend)


@pytest.mark.unit
class TestSkills:
    def test_with_skills_deep_mounts_route_and_forwards_paths(self, tmp_path):
        """``/skills/`` route mounted on FS backend; ``skills=`` forwarded."""
        from deepagents.backends import FilesystemBackend

        from muffin_agent.utils.agent_builder import MuffinAgentBuilder

        paths = ["/skills/valuation/"]
        with patch(_DEEP_PATCH) as mock_cda:
            (
                MuffinAgentBuilder(MagicMock())
                .with_skills(paths, skills_root=tmp_path)
                .build_deep_agent()
            )

        kwargs = _deep_kwargs(mock_cda)
        assert kwargs["skills"] == paths
        composite = kwargs["backend"](MagicMock())
        assert isinstance(composite.routes["/skills/"], FilesystemBackend)

    def test_with_skills_filter_middleware_appended(self, tmp_path):
        """``filter_middleware`` is appended to the middleware stack."""
        from langchain.agents.middleware.types import AgentMiddleware

        from muffin_agent.utils.agent_builder import MuffinAgentBuilder

        sentinel = MagicMock(spec=AgentMiddleware)
        with patch(_DEEP_PATCH) as mock_cda:
            (
                MuffinAgentBuilder(MagicMock())
                .with_skills(
                    ["/skills/valuation/"],
                    skills_root=tmp_path,
                    filter_middleware=sentinel,
                )
                .build_deep_agent()
            )

        assert sentinel in _deep_kwargs(mock_cda)["middleware"]

    def test_with_skills_filter_middleware_optional(self, tmp_path):
        """Omitting ``filter_middleware`` does not add extra middleware."""
        from muffin_agent.middlewares import (
            ToolErrorHandlerMiddleware,
            ToolResultCacheMiddleware,
        )
        from muffin_agent.utils.agent_builder import MuffinAgentBuilder

        with patch(_DEEP_PATCH) as mock_cda:
            (
                MuffinAgentBuilder(MagicMock())
                .with_skills(["/skills/valuation/"], skills_root=tmp_path)
                .build_deep_agent()
            )

        mw = _deep_kwargs(mock_cda)["middleware"]
        # Only the two universal middlewares, no filter middleware.
        assert len(mw) == 2
        assert isinstance(mw[0], ToolErrorHandlerMiddleware)
        assert isinstance(mw[1], ToolResultCacheMiddleware)

    def test_with_skills_called_twice_raises(self, tmp_path):
        """Calling ``with_skills`` twice raises ``ValueError``."""
        from muffin_agent.utils.agent_builder import MuffinAgentBuilder

        builder = MuffinAgentBuilder(MagicMock()).with_skills(
            ["/skills/valuation/"], skills_root=tmp_path
        )
        with pytest.raises(ValueError, match="at most once"):
            builder.with_skills(["/skills/other/"], skills_root=tmp_path)

    def test_with_skills_react_raises(self, tmp_path):
        """``build_react_agent`` after ``with_skills`` raises ``ValueError``."""
        from muffin_agent.utils.agent_builder import MuffinAgentBuilder

        builder = MuffinAgentBuilder(MagicMock()).with_skills(
            ["/skills/valuation/"], skills_root=tmp_path
        )
        with pytest.raises(ValueError, match="build_deep_agent"):
            builder.build_react_agent()


@pytest.mark.unit
class TestSubagents:
    def test_with_subagents_forwards_list(self):
        """Subagents list is forwarded to ``create_deep_agent``."""
        from muffin_agent.utils.agent_builder import MuffinAgentBuilder

        subs = [MagicMock(name="sub1"), MagicMock(name="sub2")]
        with patch(_DEEP_PATCH) as mock_cda:
            MuffinAgentBuilder(MagicMock()).with_subagents(subs).build_deep_agent()

        assert _deep_kwargs(mock_cda)["subagents"] == subs

    def test_with_subagents_react_raises(self):
        """``build_react_agent`` after ``with_subagents`` raises ``ValueError``."""
        from muffin_agent.utils.agent_builder import MuffinAgentBuilder

        builder = MuffinAgentBuilder(MagicMock()).with_subagents([MagicMock()])
        with pytest.raises(ValueError, match="build_deep_agent"):
            builder.build_react_agent()


@pytest.mark.unit
class TestTools:
    def test_with_tool_cacheable_by_default(self):
        """One cacheable tool populates the cache middleware whitelist."""
        from muffin_agent.middlewares import ToolResultCacheMiddleware
        from muffin_agent.utils.agent_builder import MuffinAgentBuilder

        tool = MagicMock(name="tool")
        tool.name = "foo"
        with patch(_DEEP_PATCH) as mock_cda:
            MuffinAgentBuilder(MagicMock()).with_tool(tool).build_deep_agent()

        kwargs = _deep_kwargs(mock_cda)
        assert kwargs["tools"] == [tool]
        cache_mw = next(
            m for m in kwargs["middleware"] if isinstance(m, ToolResultCacheMiddleware)
        )
        assert cache_mw._cacheable_tools == frozenset({"foo"})

    def test_with_tool_is_cacheable_false_excludes_from_whitelist(self):
        """``is_cacheable=False`` keeps the tool but skips caching."""
        from muffin_agent.middlewares import ToolResultCacheMiddleware
        from muffin_agent.utils.agent_builder import MuffinAgentBuilder

        tool = MagicMock(name="tool")
        tool.name = "foo"
        with patch(_DEEP_PATCH) as mock_cda:
            (
                MuffinAgentBuilder(MagicMock())
                .with_tool(tool, is_cacheable=False)
                .build_deep_agent()
            )

        kwargs = _deep_kwargs(mock_cda)
        assert kwargs["tools"] == [tool]
        cache_mw = next(
            m for m in kwargs["middleware"] if isinstance(m, ToolResultCacheMiddleware)
        )
        assert cache_mw._cacheable_tools == frozenset()

    def test_with_tool_mixed(self):
        """Mix of cacheable and non-cacheable populates only the cacheable names."""
        from muffin_agent.middlewares import ToolResultCacheMiddleware
        from muffin_agent.utils.agent_builder import MuffinAgentBuilder

        t1 = MagicMock()
        t1.name = "a"
        t2 = MagicMock()
        t2.name = "b"
        t3 = MagicMock()
        t3.name = "c"
        with patch(_DEEP_PATCH) as mock_cda:
            (
                MuffinAgentBuilder(MagicMock())
                .with_tool(t1)
                .with_tool(t2)
                .with_tool(t3, is_cacheable=False)
                .build_deep_agent()
            )

        cache_mw = next(
            m
            for m in _deep_kwargs(mock_cda)["middleware"]
            if isinstance(m, ToolResultCacheMiddleware)
        )
        assert cache_mw._cacheable_tools == frozenset({"a", "b"})

    def test_with_tool_none_cacheable_produces_empty_frozenset(self):
        """All tools non-cacheable → cache middleware with empty frozenset."""
        from muffin_agent.middlewares import ToolResultCacheMiddleware
        from muffin_agent.utils.agent_builder import MuffinAgentBuilder

        t = MagicMock()
        t.name = "x"
        with patch(_DEEP_PATCH) as mock_cda:
            (
                MuffinAgentBuilder(MagicMock())
                .with_tool(t, is_cacheable=False)
                .build_deep_agent()
            )

        cache_mw = next(
            m
            for m in _deep_kwargs(mock_cda)["middleware"]
            if isinstance(m, ToolResultCacheMiddleware)
        )
        assert cache_mw._cacheable_tools == frozenset()


@pytest.mark.unit
class TestPermissionsAndMiddleware:
    def test_with_permission_accumulates(self):
        """Permissions accumulate and are forwarded as ``permissions=``."""
        from muffin_agent.utils.agent_builder import MuffinAgentBuilder

        p1 = FilesystemPermission(operations=["write"], paths=["/**"], mode="deny")
        p2 = FilesystemPermission(operations=["read"], paths=["/workspace/**"])
        with patch(_DEEP_PATCH) as mock_cda:
            (
                MuffinAgentBuilder(MagicMock())
                .with_permission(p1)
                .with_permission(p2)
                .build_deep_agent()
            )

        assert _deep_kwargs(mock_cda)["permissions"] == [p1, p2]

    def test_permissions_none_when_empty(self):
        """``permissions`` kwarg is ``None`` when none registered."""
        from muffin_agent.utils.agent_builder import MuffinAgentBuilder

        with patch(_DEEP_PATCH) as mock_cda:
            MuffinAgentBuilder(MagicMock()).build_deep_agent()

        assert _deep_kwargs(mock_cda)["permissions"] is None

    def test_with_permission_react_raises(self):
        """``build_react_agent`` after ``with_permission`` raises ``ValueError``."""
        from muffin_agent.utils.agent_builder import MuffinAgentBuilder

        p = FilesystemPermission(operations=["write"], paths=["/**"], mode="deny")
        builder = MuffinAgentBuilder(MagicMock()).with_permission(p)
        with pytest.raises(ValueError, match="build_deep_agent"):
            builder.build_react_agent()

    def test_with_middleware_appended_after_defaults(self):
        """Caller middleware lands after universal defaults."""
        from langchain.agents.middleware.types import AgentMiddleware

        from muffin_agent.middlewares import (
            ToolErrorHandlerMiddleware,
            ToolResultCacheMiddleware,
        )
        from muffin_agent.utils.agent_builder import MuffinAgentBuilder

        x = MagicMock(spec=AgentMiddleware)
        y = MagicMock(spec=AgentMiddleware)
        with patch(_DEEP_PATCH) as mock_cda:
            (
                MuffinAgentBuilder(MagicMock())
                .with_middleware(x)
                .with_middleware(y)
                .build_deep_agent()
            )

        mw = _deep_kwargs(mock_cda)["middleware"]
        assert isinstance(mw[0], ToolErrorHandlerMiddleware)
        assert isinstance(mw[1], ToolResultCacheMiddleware)
        assert mw[-2] is x
        assert mw[-1] is y


@pytest.mark.unit
class TestMiddlewareOrder:
    def test_react_order(self):
        """Order: ToolErrorHandler, Cache, Filesystem, Memory, caller."""
        from deepagents.middleware.filesystem import FilesystemMiddleware
        from deepagents.middleware.memory import MemoryMiddleware
        from langchain.agents.middleware.types import AgentMiddleware

        from muffin_agent.middlewares import (
            ToolErrorHandlerMiddleware,
            ToolResultCacheMiddleware,
        )
        from muffin_agent.utils.agent_builder import MuffinAgentBuilder

        caller_mw = MagicMock(spec=AgentMiddleware)
        with patch(_REACT_PATCH) as mock_ca:
            (
                MuffinAgentBuilder(MagicMock())
                .with_short_term_memory()
                .with_persistent_memory()
                .with_middleware(caller_mw)
                .build_react_agent()
            )

        mw = _react_kwargs(mock_ca)["middleware"]
        assert isinstance(mw[0], ToolErrorHandlerMiddleware)
        assert isinstance(mw[1], ToolResultCacheMiddleware)
        assert isinstance(mw[2], FilesystemMiddleware)
        assert isinstance(mw[3], MemoryMiddleware)
        assert mw[4] is caller_mw


@pytest.mark.unit
class TestSystemPrompt:
    def test_with_system_prompt_template_renders(self):
        """Template rendering hides the direct ``render_template`` call."""
        from muffin_agent.utils.agent_builder import MuffinAgentBuilder

        rendered = "RENDERED"
        with (
            patch(_DEEP_PATCH) as mock_cda,
            patch("muffin_agent.utils.agent_builder.render_template") as mock_rt,
        ):
            mock_rt.return_value = rendered
            (
                MuffinAgentBuilder(MagicMock())
                .with_system_prompt_template("foo.jinja", bar=1)
                .build_deep_agent()
            )

        mock_rt.assert_any_call("foo.jinja", bar=1)
        assert _deep_kwargs(mock_cda)["system_prompt"] == rendered

    def test_with_system_prompt_overrides_template(self):
        """Last call wins between ``with_system_prompt`` and ``_template``."""
        from muffin_agent.utils.agent_builder import MuffinAgentBuilder

        with (
            patch(_DEEP_PATCH) as mock_cda,
            patch("muffin_agent.utils.agent_builder.render_template") as mock_rt,
        ):
            mock_rt.return_value = "from-template"
            (
                MuffinAgentBuilder(MagicMock())
                .with_system_prompt_template("foo.jinja")
                .with_system_prompt("raw-prompt")
                .build_deep_agent()
            )

        assert _deep_kwargs(mock_cda)["system_prompt"] == "raw-prompt"

    def test_with_system_prompt_accepts_system_message(self):
        """``with_system_prompt`` accepts ``SystemMessage`` and forwards it."""
        from langchain_core.messages import SystemMessage

        from muffin_agent.utils.agent_builder import MuffinAgentBuilder

        msg = SystemMessage(content="you are an analyst")
        with patch(_DEEP_PATCH) as mock_cda:
            MuffinAgentBuilder(MagicMock()).with_system_prompt(msg).build_deep_agent()

        result = _deep_kwargs(mock_cda)["system_prompt"]
        assert isinstance(result, SystemMessage)
        assert result.content == "you are an analyst"

    def test_system_message_augmented_with_partials(self):
        """Partials extend ``SystemMessage.content`` while preserving the type."""
        from langchain_core.messages import SystemMessage

        from muffin_agent.utils.agent_builder import MuffinAgentBuilder

        def fake_render(name, **_vars):
            return f"<{name}>"

        msg = SystemMessage(content="BASE")
        with (
            patch(_DEEP_PATCH) as mock_cda,
            patch(
                "muffin_agent.utils.agent_builder.render_template",
                side_effect=fake_render,
            ),
        ):
            (
                MuffinAgentBuilder(MagicMock())
                .with_system_prompt(msg)
                .with_sandbox()
                .build_deep_agent()
            )

        result = _deep_kwargs(mock_cda)["system_prompt"]
        assert isinstance(result, SystemMessage)
        assert result.content.startswith("BASE")
        assert "<sandbox.jinja>" in result.content

    def test_system_prompt_augmented_with_partials(self):
        """Capability partials are appended after the caller prompt."""
        from muffin_agent.utils.agent_builder import MuffinAgentBuilder

        def fake_render(name, **_vars):
            return {"base.jinja": "BASE", "sandbox.jinja": "SANDBOX_PARTIAL"}.get(
                name, f"<{name}>"
            )

        with (
            patch(_DEEP_PATCH) as mock_cda,
            patch(
                "muffin_agent.utils.agent_builder.render_template",
                side_effect=fake_render,
            ),
        ):
            (
                MuffinAgentBuilder(MagicMock())
                .with_system_prompt_template("base.jinja")
                .with_sandbox()
                .with_persistent_memory()
                .build_deep_agent()
            )

        prompt = _deep_kwargs(mock_cda)["system_prompt"]
        assert prompt.startswith("BASE")
        assert "SANDBOX_PARTIAL" in prompt
        assert "<middlewares/persistent_memory.jinja>" in prompt

    def test_partials_deduplicated(self):
        """Adding multiple tools registers the cache partial only once."""
        from muffin_agent.utils.agent_builder import MuffinAgentBuilder

        rendered: list[str] = []

        def fake_render(name, **_vars):
            rendered.append(name)
            return f"<{name}>"

        t1 = MagicMock()
        t1.name = "a"
        t2 = MagicMock()
        t2.name = "b"
        with (
            patch(_DEEP_PATCH),
            patch(
                "muffin_agent.utils.agent_builder.render_template",
                side_effect=fake_render,
            ),
        ):
            MuffinAgentBuilder(MagicMock()).with_tool(t1).with_tool(
                t2
            ).build_deep_agent()

        assert rendered.count("middlewares/tool_result_cache.jinja") == 1

    def test_no_prompt_and_no_partials_passes_none(self):
        """Bare builder passes ``system_prompt=None``."""
        from muffin_agent.utils.agent_builder import MuffinAgentBuilder

        with patch(_DEEP_PATCH) as mock_cda:
            MuffinAgentBuilder(MagicMock()).build_deep_agent()

        assert _deep_kwargs(mock_cda)["system_prompt"] is None


@pytest.mark.unit
class TestMiscKwargs:
    def test_response_format_forwarded(self):
        from muffin_agent.utils.agent_builder import MuffinAgentBuilder

        rf = MagicMock(name="rf")
        with patch(_DEEP_PATCH) as mock_cda:
            MuffinAgentBuilder(MagicMock()).with_response_format(rf).build_deep_agent()

        assert _deep_kwargs(mock_cda)["response_format"] is rf

    def test_store_forwarded(self):
        from muffin_agent.utils.agent_builder import MuffinAgentBuilder

        store = MagicMock(name="store")
        with patch(_DEEP_PATCH) as mock_cda:
            MuffinAgentBuilder(MagicMock()).with_store(store).build_deep_agent()

        assert _deep_kwargs(mock_cda)["store"] is store

    def test_context_schema_forwarded(self):
        from muffin_agent.utils.agent_builder import MuffinAgentBuilder

        schema = MagicMock(name="schema")
        with patch(_DEEP_PATCH) as mock_cda:
            MuffinAgentBuilder(MagicMock()).with_context_schema(
                schema
            ).build_deep_agent()

        assert _deep_kwargs(mock_cda)["context_schema"] is schema

    def test_checkpointer_forwarded(self):
        from muffin_agent.utils.agent_builder import MuffinAgentBuilder

        cp = MagicMock(name="checkpointer")
        with patch(_DEEP_PATCH) as mock_cda:
            MuffinAgentBuilder(MagicMock()).with_checkpointer(cp).build_deep_agent()

        assert _deep_kwargs(mock_cda)["checkpointer"] is cp


@pytest.mark.unit
class TestChaining:
    def test_method_chaining_returns_self(self):
        """Every ``with_*`` method returns the builder instance."""
        from langchain.agents.middleware.types import AgentMiddleware

        from muffin_agent.utils.agent_builder import MuffinAgentBuilder

        builder = MuffinAgentBuilder(MagicMock())
        assert builder.with_sandbox() is builder
        assert builder.with_short_term_memory() is builder
        assert builder.with_persistent_memory() is builder
        tool = MagicMock()
        tool.name = "t"
        assert builder.with_tool(tool) is builder
        assert builder.with_middleware(MagicMock(spec=AgentMiddleware)) is builder
        p = FilesystemPermission(operations=["write"], paths=["/**"], mode="deny")
        assert builder.with_permission(p) is builder
        assert builder.with_response_format(MagicMock()) is builder
        assert builder.with_store(MagicMock()) is builder
        assert builder.with_context_schema(MagicMock()) is builder
        assert builder.with_checkpointer(MagicMock()) is builder
        assert builder.with_system_prompt("x") is builder

    def test_build_twice_produces_two_agents(self):
        """The builder can be reused — two calls emit two equivalent agents."""
        from muffin_agent.utils.agent_builder import MuffinAgentBuilder

        builder = MuffinAgentBuilder(MagicMock()).with_sandbox()
        with patch(_DEEP_PATCH) as mock_cda:
            builder.build_deep_agent()
            builder.build_deep_agent()

        assert mock_cda.call_count == 2
        # Middleware lists are freshly assembled per build — distinct objects.
        _, first = mock_cda.call_args_list[0]
        _, second = mock_cda.call_args_list[1]
        assert first["middleware"] is not second["middleware"]
