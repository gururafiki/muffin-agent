"""Integration tests for the subagent-refinement middlewares."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import HumanMessage, SystemMessage

from muffin_agent.middlewares.subagent_refinement import (
    CollectionFindings,
    Gap,
    GapReason,
    SubagentRefinementMiddleware,
    SubagentRefinementParentMiddleware,
    child_instructions,
    parent_instructions,
)


def _backend_with_prior(
    findings_json: str | None,
) -> tuple[MagicMock, MagicMock, MagicMock]:
    backend = MagicMock()
    if findings_json is not None:
        backend.aread = AsyncMock(
            return_value=MagicMock(
                error=None,
                file_data={"content": findings_json, "encoding": "utf-8"},
            )
        )
    else:
        backend.aread = AsyncMock(
            return_value=MagicMock(error="not found", file_data=None)
        )
    backend.awrite = AsyncMock()
    factory = MagicMock(return_value=backend)
    return backend, factory, MagicMock()  # backend, factory, runtime


# ── Child middleware ────────────────────────────────────────────────


@pytest.mark.unit
class TestBeforeAgent:
    @pytest.mark.asyncio
    async def test_no_marker_in_description_returns_none(self):
        _, factory, runtime = _backend_with_prior(None)
        mw = SubagentRefinementMiddleware(backend_factory=factory)
        state = {"messages": [HumanMessage(content="just collect data")]}

        result = await mw.abefore_agent(state, runtime)
        assert result is None
        factory.assert_not_called()

    @pytest.mark.asyncio
    async def test_marker_loads_prior_findings_into_state(self):
        prior = CollectionFindings(
            call_id="abc",
            obtained={"pe": 12.3},
            gaps=[Gap(field="ev_ebitda", reason=GapReason.NO_DATA)],
        )
        _, factory, runtime = _backend_with_prior(prior.model_dump_json())
        mw = SubagentRefinementMiddleware(backend_factory=factory)
        state = {
            "messages": [
                HumanMessage(content="prior_call_id=abc fill ev_ebitda please")
            ]
        }

        result = await mw.abefore_agent(state, runtime)
        assert result is not None
        assert result["prior_findings"]["call_id"] == "abc"

    @pytest.mark.asyncio
    async def test_marker_but_missing_file_returns_none(self):
        _, factory, runtime = _backend_with_prior(None)  # backend errors
        mw = SubagentRefinementMiddleware(backend_factory=factory)
        state = {"messages": [HumanMessage(content="prior_call_id=abc")]}

        assert await mw.abefore_agent(state, runtime) is None


@pytest.mark.unit
class TestChildWrapModelCall:
    @pytest.mark.asyncio
    async def test_amends_with_static_rules_when_no_prior(self):
        """Child always prepends the static rules even without prior data."""
        mw = SubagentRefinementMiddleware(backend_factory=MagicMock())

        captured: dict = {}

        async def handler(req):
            captured["system"] = req.system_message
            return MagicMock()

        request = MagicMock()
        request.state = {}
        request.system_message = SystemMessage(content="base")
        request.override = MagicMock(
            side_effect=lambda **kw: SimpleNamespace(
                system_message=kw["system_message"], state=request.state
            )
        )

        await mw.awrap_model_call(request, handler)
        sys = captured["system"]
        assert isinstance(sys, SystemMessage)
        assert "base" in sys.content
        # Static rules from prompts.py are amended onto the system prompt.
        assert child_instructions().split("\n", 1)[0] in sys.content

    @pytest.mark.asyncio
    async def test_amends_with_prior_block_when_state_carries_findings(self):
        prior = CollectionFindings(call_id="abc", obtained={"pe": 12.3})
        mw = SubagentRefinementMiddleware(backend_factory=MagicMock())

        captured: dict = {}

        async def handler(req):
            captured["system"] = req.system_message
            return MagicMock()

        request = MagicMock()
        request.state = {"prior_findings": prior.model_dump()}
        request.system_message = SystemMessage(content="base")
        request.override = MagicMock(
            side_effect=lambda **kw: SimpleNamespace(
                system_message=kw["system_message"], state=request.state
            )
        )

        await mw.awrap_model_call(request, handler)
        sys = captured["system"]
        assert "base" in sys.content
        # Both the rules and the per-call data are present.
        assert child_instructions().split("\n", 1)[0] in sys.content
        # The rendered prior block uses ``## Prior call context`` as a header.
        assert "## Prior call context" in sys.content
        assert "call_id=abc" in sys.content

    @pytest.mark.asyncio
    async def test_malformed_prior_falls_back_to_rules_only(self):
        mw = SubagentRefinementMiddleware(backend_factory=MagicMock())

        async def handler(req):
            return req

        request = MagicMock()
        request.state = {"prior_findings": {"not_a_valid": "schema"}}
        request.system_message = SystemMessage(content="base")
        request.override = MagicMock(
            side_effect=lambda **kw: SimpleNamespace(
                system_message=kw["system_message"], state=request.state
            )
        )

        result = await mw.awrap_model_call(request, handler)
        sys = result.system_message
        # Rules still present, but no rendered prior block.
        assert child_instructions().split("\n", 1)[0] in sys.content
        assert "## Prior call context" not in sys.content


@pytest.mark.unit
class TestAfterAgent:
    @pytest.mark.asyncio
    async def test_writes_findings_to_scratch(self):
        backend, factory, runtime = _backend_with_prior(None)
        findings = CollectionFindings(call_id="abc", obtained={"pe": 12.3})
        mw = SubagentRefinementMiddleware(backend_factory=factory)

        result = await mw.aafter_agent(
            state={"structured_response": findings}, runtime=runtime
        )

        backend.awrite.assert_awaited_once()
        path, payload = backend.awrite.call_args.args
        assert path == "/scratch/subagent_runs/abc.json"
        # Echoes findings back so the parent can cite the call_id.
        assert isinstance(result, dict)
        assert result["structured_response"].call_id == "abc"
        # Sanity: payload is parseable.
        assert json.loads(payload)["call_id"] == "abc"

    @pytest.mark.asyncio
    async def test_populates_call_id_when_missing(self):
        backend, factory, runtime = _backend_with_prior(None)
        findings = CollectionFindings(call_id="", obtained={"x": 1})
        mw = SubagentRefinementMiddleware(backend_factory=factory)

        result = await mw.aafter_agent(
            state={"structured_response": findings}, runtime=runtime
        )

        new_call_id = result["structured_response"].call_id
        assert new_call_id and new_call_id != ""

    @pytest.mark.asyncio
    async def test_no_structured_response_is_noop(self):
        _, factory, runtime = _backend_with_prior(None)
        mw = SubagentRefinementMiddleware(backend_factory=factory)
        assert await mw.aafter_agent(state={}, runtime=runtime) is None

    @pytest.mark.asyncio
    async def test_non_collection_findings_response_is_noop(self):
        _, factory, runtime = _backend_with_prior(None)
        mw = SubagentRefinementMiddleware(backend_factory=factory)
        # Defensive path: someone overrode response_format with a dict.
        assert (
            await mw.aafter_agent(
                state={"structured_response": {"call_id": "abc"}}, runtime=runtime
            )
            is None
        )


# ── Parent middleware ───────────────────────────────────────────────


@pytest.mark.unit
class TestParentMiddleware:
    def test_has_no_state_or_tools(self):
        mw = SubagentRefinementParentMiddleware()
        assert mw.tools == []
        # No state schema attribute means it inherits the default AgentState.
        assert mw.state_schema is not None

    @pytest.mark.asyncio
    async def test_amends_system_message_with_parent_rules(self):
        mw = SubagentRefinementParentMiddleware()

        captured: dict = {}

        async def handler(req):
            captured["system"] = req.system_message
            return MagicMock()

        request = MagicMock()
        request.system_message = SystemMessage(content="orchestrator base")
        request.override = MagicMock(
            side_effect=lambda **kw: SimpleNamespace(
                system_message=kw["system_message"]
            )
        )

        await mw.awrap_model_call(request, handler)
        sys = captured["system"]
        assert isinstance(sys, SystemMessage)
        assert "orchestrator base" in sys.content
        assert parent_instructions().split("\n", 1)[0] in sys.content

    @pytest.mark.asyncio
    async def test_amends_when_no_prior_system_message(self):
        mw = SubagentRefinementParentMiddleware()

        captured: dict = {}

        async def handler(req):
            captured["system"] = req.system_message
            return MagicMock()

        request = MagicMock()
        request.system_message = None
        request.override = MagicMock(
            side_effect=lambda **kw: SimpleNamespace(
                system_message=kw["system_message"]
            )
        )

        await mw.awrap_model_call(request, handler)
        sys = captured["system"]
        assert isinstance(sys, SystemMessage)
        assert parent_instructions().split("\n", 1)[0] in sys.content
