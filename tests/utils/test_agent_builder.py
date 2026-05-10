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
        from langchain.agents.middleware import (
            ModelRetryMiddleware,
            ToolRetryMiddleware,
        )

        from muffin_agent.middlewares import (
            ToolKnowledgeMiddleware,
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
        assert len(mw) == 4
        assert isinstance(mw[0], ModelRetryMiddleware)
        assert isinstance(mw[1], ToolKnowledgeMiddleware)
        assert isinstance(mw[2], ToolResultCacheMiddleware)
        assert mw[2]._cacheable_tools is None
        assert isinstance(mw[3], ToolRetryMiddleware)

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
        from langchain.agents.middleware import (
            ModelRetryMiddleware,
            ToolRetryMiddleware,
        )

        from muffin_agent.middlewares import (
            ToolKnowledgeMiddleware,
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
        # Only the four universal middlewares, no filter middleware.
        assert len(mw) == 4
        assert isinstance(mw[0], ModelRetryMiddleware)
        assert isinstance(mw[1], ToolKnowledgeMiddleware)
        assert isinstance(mw[2], ToolResultCacheMiddleware)
        assert isinstance(mw[3], ToolRetryMiddleware)

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
        from langchain.agents.middleware import ModelRetryMiddleware
        from langchain.agents.middleware.types import AgentMiddleware

        from muffin_agent.middlewares import (
            ToolKnowledgeMiddleware,
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
        assert isinstance(mw[0], ModelRetryMiddleware)
        assert isinstance(mw[1], ToolKnowledgeMiddleware)
        assert isinstance(mw[2], ToolResultCacheMiddleware)
        assert mw[-2] is x
        assert mw[-1] is y


@pytest.mark.unit
class TestToolRetryWiring:
    def test_tool_retry_universal_with_muffin_defaults(self):
        """``ToolRetryMiddleware`` is wired with muffin defaults."""
        from langchain.agents.middleware import ToolRetryMiddleware

        from muffin_agent.utils.agent_builder import MuffinAgentBuilder

        with patch(_DEEP_PATCH) as mock_cda:
            MuffinAgentBuilder(MagicMock()).build_deep_agent()

        mw = _deep_kwargs(mock_cda)["middleware"]
        retry = next(m for m in mw if isinstance(m, ToolRetryMiddleware))
        assert retry.max_retries == 1
        assert retry.on_failure == "continue"
        assert retry.initial_delay == 1.0
        assert retry.max_delay == 10.0
        assert retry.backoff_factor == 2.0
        assert retry.jitter is True

    def test_filter_retries_on_5xx_tool_exceptions(self):
        """The filter accepts HTTP 5xx / gateway / network errors only."""
        from langchain_core.tools import ToolException

        from muffin_agent.utils.agent_builder import _should_retry_tool_call

        # Transient — should retry.
        for body in (
            "Error calling tool 'X': HTTP error 502: Bad Gateway",
            "Error calling tool 'Y': HTTP error 503: Service Unavailable",
            "Error calling tool 'Z': HTTP error 504: Gateway Timeout",
            "Error calling tool 'A': HTTP error 500: Internal Server Error",
            "Connection error to MCP server",
            "Request timed out after 30s",
        ):
            assert _should_retry_tool_call(ToolException(body)) is True, body

        # Permanent — should NOT retry.
        for body in (
            "HTTP error 422: Unprocessable Entity ... limit must be ≤ 5",
            "HTTP error 400: Bad Request ... Missing credential 'intrinio_api_key'",
            "HTTP error 401: Unauthorized",
            "HTTP error 404: Not Found",
            "No estimates data was returned for: AMZN",
        ):
            assert _should_retry_tool_call(ToolException(body)) is False, body

    def test_filter_retries_network_errors(self):
        """Plain TimeoutError / ConnectionError also retry."""
        from muffin_agent.utils.agent_builder import _should_retry_tool_call

        assert _should_retry_tool_call(TimeoutError("read timeout")) is True
        assert _should_retry_tool_call(ConnectionError("conn reset")) is True

    def test_filter_does_not_retry_unrelated_exceptions(self):
        """``ValueError`` / ``RuntimeError`` etc. propagate without retry."""
        from muffin_agent.utils.agent_builder import _should_retry_tool_call

        assert _should_retry_tool_call(ValueError("boom")) is False
        assert _should_retry_tool_call(RuntimeError("nope")) is False


@pytest.mark.unit
class TestSubagentRefinementWiring:
    def test_default_omits_middleware_and_partial(self):
        """Without ``with_subagent_refinement`` no middleware/partial is added."""
        from muffin_agent.middlewares import SubagentRefinementMiddleware
        from muffin_agent.utils.agent_builder import MuffinAgentBuilder

        with patch(_DEEP_PATCH) as mock_cda:
            MuffinAgentBuilder(MagicMock()).build_deep_agent()

        kwargs = _deep_kwargs(mock_cda)
        assert not any(
            isinstance(m, SubagentRefinementMiddleware) for m in kwargs["middleware"]
        )
        # Default response_format is None.
        assert kwargs["response_format"] is None

    def test_child_role_wires_middleware_and_response_format(self):
        """Subagent (no subagents wired) gets middleware + AutoStrategy."""
        from langchain.agents.structured_output import AutoStrategy

        from muffin_agent.middlewares import (
            CollectionFindings,
            SubagentRefinementMiddleware,
        )
        from muffin_agent.utils.agent_builder import MuffinAgentBuilder

        with patch(_REACT_PATCH) as mock_ca:
            (
                MuffinAgentBuilder(MagicMock())
                .with_short_term_memory()  # backend route required
                .with_subagent_refinement()
                .build_react_agent()
            )

        kwargs = _react_kwargs(mock_ca)
        mw = kwargs["middleware"]
        assert any(isinstance(m, SubagentRefinementMiddleware) for m in mw)
        rf = kwargs["response_format"]
        assert isinstance(rf, AutoStrategy)
        assert rf.schema is CollectionFindings

    def test_parent_role_wires_parent_middleware(self):
        """Parent (subagents wired) gets the parent middleware, not the child."""
        from deepagents import CompiledSubAgent

        from muffin_agent.middlewares import (
            SubagentRefinementMiddleware,
            SubagentRefinementParentMiddleware,
        )
        from muffin_agent.utils.agent_builder import MuffinAgentBuilder

        sub = CompiledSubAgent(
            name="x",
            description="x",
            runnable=MagicMock(),
        )
        with patch(_DEEP_PATCH) as mock_cda:
            (
                MuffinAgentBuilder(MagicMock())
                .with_subagents([sub])
                .with_subagent_refinement()
                .build_deep_agent()
            )

        kwargs = _deep_kwargs(mock_cda)
        mw = kwargs["middleware"]
        # Parent class is registered; child class is not.
        assert any(isinstance(m, SubagentRefinementParentMiddleware) for m in mw)
        assert not any(isinstance(m, SubagentRefinementMiddleware) for m in mw)
        # The agent_builder no longer touches the system_prompt for refinement.
        assert kwargs["system_prompt"] is None

    def test_caller_response_format_wins(self):
        """An explicit ``with_response_format`` is preserved."""
        from muffin_agent.utils.agent_builder import MuffinAgentBuilder

        sentinel = object()
        with patch(_REACT_PATCH) as mock_ca:
            (
                MuffinAgentBuilder(MagicMock())
                .with_short_term_memory()
                .with_response_format(sentinel)  # type: ignore[arg-type]
                .with_subagent_refinement()
                .build_react_agent()
            )

        assert _react_kwargs(mock_ca)["response_format"] is sentinel


@pytest.mark.unit
class TestModelRetryWiring:
    def test_wired_with_filter_and_on_failure_error(self):
        """Retry middleware is wired with the right filter and propagates errors."""
        import httpx
        import openai
        from langchain.agents.middleware import ModelRetryMiddleware

        from muffin_agent.utils.agent_builder import MuffinAgentBuilder

        with patch(_DEEP_PATCH) as mock_cda:
            MuffinAgentBuilder(MagicMock()).build_deep_agent()

        mw = _deep_kwargs(mock_cda)["middleware"]
        retry_mw = next(m for m in mw if isinstance(m, ModelRetryMiddleware))

        assert retry_mw.max_retries == 3
        assert retry_mw.on_failure == "error"
        assert retry_mw.backoff_factor == 2.0
        assert retry_mw.initial_delay == 1.0
        assert retry_mw.max_delay == 30.0
        assert retry_mw.jitter is True

        retry_on = retry_mw.retry_on
        assert callable(retry_on)
        request = httpx.Request("POST", "https://example.test/v1")
        api_err = openai.APIError("Provider returned error", request=request, body=None)
        auth_err = openai.AuthenticationError(
            "bad key", response=httpx.Response(401, request=request), body=None
        )
        assert retry_on(api_err) is True
        assert retry_on(auth_err) is False
        assert retry_on(ValueError("unrelated")) is False


@pytest.mark.unit
class TestModelFallbackWiring:
    def test_no_fallback_models_means_no_fallback_middleware(self):
        """Without ``with_fallback_models`` the fallback middleware is absent."""
        from langchain.agents.middleware import ModelFallbackMiddleware

        from muffin_agent.utils.agent_builder import MuffinAgentBuilder

        with patch(_DEEP_PATCH) as mock_cda:
            MuffinAgentBuilder(MagicMock()).build_deep_agent()

        mw = _deep_kwargs(mock_cda)["middleware"]
        assert not any(isinstance(m, ModelFallbackMiddleware) for m in mw)

    def test_with_fallback_models_wires_outermost(self):
        """``with_fallback_models`` registers ``ModelFallbackMiddleware`` first."""
        from langchain.agents.middleware import (
            ModelFallbackMiddleware,
            ModelRetryMiddleware,
        )

        from muffin_agent.utils.agent_builder import MuffinAgentBuilder

        primary = MagicMock(name="primary")
        fb1 = MagicMock(name="fallback-1")
        fb2 = MagicMock(name="fallback-2")
        with patch(_DEEP_PATCH) as mock_cda:
            (
                MuffinAgentBuilder(primary)
                .with_fallback_models(fb1, fb2)
                .build_deep_agent()
            )

        mw = _deep_kwargs(mock_cda)["middleware"]
        assert isinstance(mw[0], ModelFallbackMiddleware)
        assert mw[0].models == [fb1, fb2]
        # Retry must remain inside the fallback boundary so retries happen
        # on the current model before fallback switches.
        assert isinstance(mw[1], ModelRetryMiddleware)

    def test_with_fallback_models_accumulates_across_calls(self):
        """Multiple ``with_fallback_models`` calls extend the chain."""
        from langchain.agents.middleware import ModelFallbackMiddleware

        from muffin_agent.utils.agent_builder import MuffinAgentBuilder

        a, b, c = MagicMock(), MagicMock(), MagicMock()
        with patch(_DEEP_PATCH) as mock_cda:
            (
                MuffinAgentBuilder(MagicMock())
                .with_fallback_models(a)
                .with_fallback_models(b, c)
                .build_deep_agent()
            )

        mw = _deep_kwargs(mock_cda)["middleware"]
        fallback = next(m for m in mw if isinstance(m, ModelFallbackMiddleware))
        assert fallback.models == [a, b, c]


@pytest.mark.unit
class TestContextEditingWiring:
    def test_default_omits_context_editing(self):
        """Without ``with_context_editing`` the middleware is absent."""
        from langchain.agents.middleware import ContextEditingMiddleware

        from muffin_agent.utils.agent_builder import MuffinAgentBuilder

        with patch(_DEEP_PATCH) as mock_cda:
            MuffinAgentBuilder(MagicMock()).build_deep_agent()

        mw = _deep_kwargs(mock_cda)["middleware"]
        assert not any(isinstance(m, ContextEditingMiddleware) for m in mw)

    def test_with_context_editing_wires_clear_tool_uses(self):
        """``with_context_editing`` registers a ``ClearToolUsesEdit`` strategy."""
        from langchain.agents.middleware import (
            ClearToolUsesEdit,
            ContextEditingMiddleware,
        )

        from muffin_agent.utils.agent_builder import MuffinAgentBuilder

        with patch(_DEEP_PATCH) as mock_cda:
            (
                MuffinAgentBuilder(MagicMock())
                .with_context_editing(trigger=12_000, keep=2)
                .build_deep_agent()
            )

        mw = _deep_kwargs(mock_cda)["middleware"]
        ce = next(m for m in mw if isinstance(m, ContextEditingMiddleware))
        assert len(ce.edits) == 1
        edit = ce.edits[0]
        assert isinstance(edit, ClearToolUsesEdit)
        assert edit.trigger == 12_000
        assert edit.keep == 2

    def test_last_with_context_editing_call_wins(self):
        """Calling twice replaces the prior configuration."""
        from langchain.agents.middleware import ContextEditingMiddleware

        from muffin_agent.utils.agent_builder import MuffinAgentBuilder

        with patch(_DEEP_PATCH) as mock_cda:
            (
                MuffinAgentBuilder(MagicMock())
                .with_context_editing(trigger=10_000, keep=1)
                .with_context_editing(trigger=20_000, keep=5)
                .build_deep_agent()
            )

        mw = _deep_kwargs(mock_cda)["middleware"]
        ce = next(m for m in mw if isinstance(m, ContextEditingMiddleware))
        assert ce.edits[0].trigger == 20_000
        assert ce.edits[0].keep == 5


@pytest.mark.unit
class TestSummarizationWiring:
    def test_default_omits_summarization(self):
        """Without ``with_summarization`` the middleware is absent."""
        from langchain.agents.middleware import SummarizationMiddleware

        from muffin_agent.utils.agent_builder import MuffinAgentBuilder

        with patch(_DEEP_PATCH) as mock_cda:
            MuffinAgentBuilder(MagicMock()).build_deep_agent()

        mw = _deep_kwargs(mock_cda)["middleware"]
        assert not any(isinstance(m, SummarizationMiddleware) for m in mw)

    def test_with_summarization_uses_primary_model_by_default(self):
        """``with_summarization()`` defaults to the agent's primary model."""
        from langchain.agents.middleware import SummarizationMiddleware

        from muffin_agent.utils.agent_builder import MuffinAgentBuilder

        primary = MagicMock(name="primary")
        with patch(_DEEP_PATCH) as mock_cda:
            MuffinAgentBuilder(primary).with_summarization().build_deep_agent()

        mw = _deep_kwargs(mock_cda)["middleware"]
        sm = next(m for m in mw if isinstance(m, SummarizationMiddleware))
        assert sm.model is primary

    def test_with_summarization_accepts_override_model(self):
        """An explicit summariser model overrides the primary."""
        from langchain.agents.middleware import SummarizationMiddleware

        from muffin_agent.utils.agent_builder import MuffinAgentBuilder

        summariser = MagicMock(name="summariser")
        with patch(_DEEP_PATCH) as mock_cda:
            (
                MuffinAgentBuilder(MagicMock(name="primary"))
                .with_summarization(summariser)
                .build_deep_agent()
            )

        mw = _deep_kwargs(mock_cda)["middleware"]
        sm = next(m for m in mw if isinstance(m, SummarizationMiddleware))
        assert sm.model is summariser


@pytest.mark.unit
class TestCallLimitWiring:
    def test_default_omits_call_limits(self):
        """Without builder methods the call-limit middleware are absent."""
        from langchain.agents.middleware import (
            ModelCallLimitMiddleware,
            ToolCallLimitMiddleware,
        )

        from muffin_agent.utils.agent_builder import MuffinAgentBuilder

        with patch(_DEEP_PATCH) as mock_cda:
            MuffinAgentBuilder(MagicMock()).build_deep_agent()

        mw = _deep_kwargs(mock_cda)["middleware"]
        assert not any(isinstance(m, ModelCallLimitMiddleware) for m in mw)
        assert not any(isinstance(m, ToolCallLimitMiddleware) for m in mw)

    def test_with_model_call_limit_wires_outermost(self):
        """``with_model_call_limit`` is registered before all other middleware."""
        from langchain.agents.middleware import ModelCallLimitMiddleware

        from muffin_agent.utils.agent_builder import MuffinAgentBuilder

        with patch(_DEEP_PATCH) as mock_cda:
            (
                MuffinAgentBuilder(MagicMock())
                .with_model_call_limit(run_limit=20, thread_limit=100)
                .build_deep_agent()
            )

        mw = _deep_kwargs(mock_cda)["middleware"]
        assert isinstance(mw[0], ModelCallLimitMiddleware)
        assert mw[0].run_limit == 20
        assert mw[0].thread_limit == 100
        assert mw[0].exit_behavior == "end"

    def test_with_tool_call_limit_accumulates_multiple(self):
        """Multiple ``with_tool_call_limit`` calls produce multiple middlewares."""
        from langchain.agents.middleware import ToolCallLimitMiddleware

        from muffin_agent.utils.agent_builder import MuffinAgentBuilder

        with patch(_DEEP_PATCH) as mock_cda:
            (
                MuffinAgentBuilder(MagicMock())
                .with_tool_call_limit(run_limit=40)  # global cap
                .with_tool_call_limit(tool_name="task", run_limit=10)
                .build_deep_agent()
            )

        mw = _deep_kwargs(mock_cda)["middleware"]
        limits = [m for m in mw if isinstance(m, ToolCallLimitMiddleware)]
        assert len(limits) == 2
        assert limits[0].tool_name is None and limits[0].run_limit == 40
        assert limits[1].tool_name == "task" and limits[1].run_limit == 10

    def test_with_tool_inline_run_limit_appends_per_tool_middleware(self):
        """``with_tool(..., run_limit=N)`` adds a per-tool call-limit middleware."""
        from langchain.agents.middleware import ToolCallLimitMiddleware

        from muffin_agent.utils.agent_builder import MuffinAgentBuilder

        my_tool = MagicMock(name="task")
        my_tool.name = "task"

        with patch(_DEEP_PATCH) as mock_cda:
            (
                MuffinAgentBuilder(MagicMock())
                .with_tool(my_tool, run_limit=10, thread_limit=50)
                .build_deep_agent()
            )

        mw = _deep_kwargs(mock_cda)["middleware"]
        limits = [m for m in mw if isinstance(m, ToolCallLimitMiddleware)]
        assert len(limits) == 1
        assert limits[0].tool_name == "task"
        assert limits[0].run_limit == 10
        assert limits[0].thread_limit == 50

    def test_with_tool_default_run_limit(self):
        """``with_tool`` registers per-tool ToolCallLimitMiddleware with run_limit=6."""
        from langchain.agents.middleware import ToolCallLimitMiddleware

        from muffin_agent.utils.agent_builder import (
            _DEFAULT_TOOL_RUN_LIMIT,
            MuffinAgentBuilder,
        )

        my_tool = MagicMock()
        my_tool.name = "noop"
        with patch(_DEEP_PATCH) as mock_cda:
            MuffinAgentBuilder(MagicMock()).with_tool(my_tool).build_deep_agent()

        mw = _deep_kwargs(mock_cda)["middleware"]
        limits = [m for m in mw if isinstance(m, ToolCallLimitMiddleware)]
        assert len(limits) == 1
        assert limits[0].tool_name == "noop"
        assert limits[0].run_limit == _DEFAULT_TOOL_RUN_LIMIT

    def test_with_tool_run_limit_none_omits_middleware(self):
        """``with_tool(run_limit=None)`` opts out of the per-tool call limit."""
        from langchain.agents.middleware import ToolCallLimitMiddleware

        from muffin_agent.utils.agent_builder import MuffinAgentBuilder

        my_tool = MagicMock()
        my_tool.name = "noop"
        with patch(_DEEP_PATCH) as mock_cda:
            MuffinAgentBuilder(MagicMock()).with_tool(
                my_tool, run_limit=None
            ).build_deep_agent()

        mw = _deep_kwargs(mock_cda)["middleware"]
        assert not any(isinstance(m, ToolCallLimitMiddleware) for m in mw)

    def test_with_tool_per_tool_and_global_limits_coexist(self):
        """Inline per-tool limits compose with a global ``with_tool_call_limit`` cap."""
        from langchain.agents.middleware import ToolCallLimitMiddleware

        from muffin_agent.utils.agent_builder import MuffinAgentBuilder

        a = MagicMock()
        a.name = "tool_a"
        b = MagicMock()
        b.name = "tool_b"
        with patch(_DEEP_PATCH) as mock_cda:
            (
                MuffinAgentBuilder(MagicMock())
                .with_tool(a, run_limit=3)
                .with_tool(b)
                .with_tool_call_limit(run_limit=20)  # global cap
                .build_deep_agent()
            )

        mw = _deep_kwargs(mock_cda)["middleware"]
        limits = [m for m in mw if isinstance(m, ToolCallLimitMiddleware)]
        from muffin_agent.utils.agent_builder import _DEFAULT_TOOL_RUN_LIMIT

        assert {(lim.tool_name, lim.run_limit) for lim in limits} == {
            ("tool_a", 3),
            ("tool_b", _DEFAULT_TOOL_RUN_LIMIT),
            (None, 20),
        }

    def test_with_tool_inline_limit_requires_tool_name(self):
        """Per-tool limits raise when the tool lacks a string ``.name``."""
        from muffin_agent.utils.agent_builder import MuffinAgentBuilder

        anonymous_tool: dict = {"type": "provider", "id": "x"}  # no `.name`
        with pytest.raises(ValueError, match="Per-tool call limits require"):
            MuffinAgentBuilder(MagicMock()).with_tool(anonymous_tool, run_limit=5)

    def test_last_with_model_call_limit_call_wins(self):
        """Calling ``with_model_call_limit`` twice keeps only the last."""
        from langchain.agents.middleware import ModelCallLimitMiddleware

        from muffin_agent.utils.agent_builder import MuffinAgentBuilder

        with patch(_DEEP_PATCH) as mock_cda:
            (
                MuffinAgentBuilder(MagicMock())
                .with_model_call_limit(run_limit=10)
                .with_model_call_limit(run_limit=25, exit_behavior="error")
                .build_deep_agent()
            )

        mw = _deep_kwargs(mock_cda)["middleware"]
        limits = [m for m in mw if isinstance(m, ModelCallLimitMiddleware)]
        assert len(limits) == 1
        assert limits[0].run_limit == 25
        assert limits[0].exit_behavior == "error"


@pytest.mark.unit
class TestMiddlewareOrder:
    def test_react_order(self):
        """ReAct stack: ModelRetry → ToolError → Cache → ToolRetry → FS → Memory."""
        from deepagents.middleware.filesystem import FilesystemMiddleware
        from deepagents.middleware.memory import MemoryMiddleware
        from langchain.agents.middleware import (
            ModelRetryMiddleware,
            ToolRetryMiddleware,
        )
        from langchain.agents.middleware.types import AgentMiddleware

        from muffin_agent.middlewares import (
            ToolKnowledgeMiddleware,
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
        assert isinstance(mw[0], ModelRetryMiddleware)
        assert isinstance(mw[1], ToolKnowledgeMiddleware)
        assert isinstance(mw[2], ToolResultCacheMiddleware)
        assert isinstance(mw[3], ToolRetryMiddleware)
        assert isinstance(mw[4], FilesystemMiddleware)
        assert isinstance(mw[5], MemoryMiddleware)
        assert mw[6] is caller_mw

    def test_full_order_with_all_optionals(self):
        """All optional middleware land in the documented order."""
        from deepagents.middleware.filesystem import FilesystemMiddleware
        from deepagents.middleware.memory import MemoryMiddleware
        from langchain.agents.middleware import (
            ContextEditingMiddleware,
            ModelFallbackMiddleware,
            ModelRetryMiddleware,
            SummarizationMiddleware,
            ToolRetryMiddleware,
        )

        from muffin_agent.middlewares import (
            ToolKnowledgeMiddleware,
            ToolResultCacheMiddleware,
        )
        from muffin_agent.utils.agent_builder import MuffinAgentBuilder

        with patch(_REACT_PATCH) as mock_ca:
            (
                MuffinAgentBuilder(MagicMock())
                .with_fallback_models(MagicMock(name="fb"))
                .with_context_editing()
                .with_summarization()
                .with_short_term_memory()
                .with_persistent_memory()
                .build_react_agent()
            )

        mw = _react_kwargs(mock_ca)["middleware"]
        assert isinstance(mw[0], ModelFallbackMiddleware)
        assert isinstance(mw[1], ModelRetryMiddleware)
        assert isinstance(mw[2], ContextEditingMiddleware)
        assert isinstance(mw[3], SummarizationMiddleware)
        assert isinstance(mw[4], ToolKnowledgeMiddleware)
        assert isinstance(mw[5], ToolResultCacheMiddleware)
        assert isinstance(mw[6], ToolRetryMiddleware)
        assert isinstance(mw[7], FilesystemMiddleware)
        assert isinstance(mw[8], MemoryMiddleware)

    def test_react_order_with_fallback_models(self):
        """Fallback is outermost when configured."""
        from deepagents.middleware.filesystem import FilesystemMiddleware
        from deepagents.middleware.memory import MemoryMiddleware
        from langchain.agents.middleware import (
            ModelFallbackMiddleware,
            ModelRetryMiddleware,
            ToolRetryMiddleware,
        )
        from langchain.agents.middleware.types import AgentMiddleware

        from muffin_agent.middlewares import (
            ToolKnowledgeMiddleware,
            ToolResultCacheMiddleware,
        )
        from muffin_agent.utils.agent_builder import MuffinAgentBuilder

        caller_mw = MagicMock(spec=AgentMiddleware)
        with patch(_REACT_PATCH) as mock_ca:
            (
                MuffinAgentBuilder(MagicMock())
                .with_fallback_models(MagicMock(name="fb"))
                .with_short_term_memory()
                .with_persistent_memory()
                .with_middleware(caller_mw)
                .build_react_agent()
            )

        mw = _react_kwargs(mock_ca)["middleware"]
        assert isinstance(mw[0], ModelFallbackMiddleware)
        assert isinstance(mw[1], ModelRetryMiddleware)
        assert isinstance(mw[2], ToolKnowledgeMiddleware)
        assert isinstance(mw[3], ToolResultCacheMiddleware)
        assert isinstance(mw[4], ToolRetryMiddleware)
        assert isinstance(mw[5], FilesystemMiddleware)
        assert isinstance(mw[6], MemoryMiddleware)
        assert mw[7] is caller_mw


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
        """with_sandbox() called twice registers the cache partial only once."""
        from muffin_agent.utils.agent_builder import MuffinAgentBuilder

        rendered: list[str] = []

        def fake_render(name, **_vars):
            rendered.append(name)
            return f"<{name}>"

        with (
            patch(_DEEP_PATCH),
            patch(
                "muffin_agent.utils.agent_builder.render_template",
                side_effect=fake_render,
            ),
        ):
            MuffinAgentBuilder(
                MagicMock()
            ).with_sandbox().with_sandbox().build_deep_agent()

        assert rendered.count("middlewares/tool_result_cache.jinja") == 1

    def test_with_tool_does_not_add_cache_partial(self):
        """with_tool() does not add the cache partial — that is with_sandbox()'s job."""
        from muffin_agent.utils.agent_builder import MuffinAgentBuilder

        rendered: list[str] = []

        def fake_render(name, **_vars):
            rendered.append(name)
            return f"<{name}>"

        t1 = MagicMock()
        t1.name = "a"
        with (
            patch(_DEEP_PATCH),
            patch(
                "muffin_agent.utils.agent_builder.render_template",
                side_effect=fake_render,
            ),
        ):
            MuffinAgentBuilder(MagicMock()).with_tool(t1).build_deep_agent()

        assert "middlewares/tool_result_cache.jinja" not in rendered

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
