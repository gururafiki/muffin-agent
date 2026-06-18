"""E2E integration tests — every persona exercises its FULL MCP tool list.

Complements the other two persona-flavoured suites, which deliberately do not
cover tool fan-out:

* ``test_council_graph_e2e.py`` runs all 13 personas in parallel but scripts each
  ``collect_data`` to emit its ``RawData`` immediately — **zero** MCP calls.
* ``test_persona_peter_lynch.py`` drives the full 3-node pipeline but calls only
  **one** of the persona's tools.

Here each persona (parametrised over ``PERSONA_BUILDERS``) runs standalone and its
``collect_data`` ReAct loop invokes **every** tool in the persona's ``_MCP_TOOLS``
list plus ``execute_python`` — issued as one parallel-tool-call turn, the way a
real tool-calling LLM batches independent fetches (and well within each persona's
``with_model_call_limit`` budget of 8–10). This proves, per persona:

* every allowlisted MCP tool resolves through the real ``get_tools`` filter and
  executes against its (schema-correct) fixture — a missing/broken fixture or a
  tool name drifting from the OpenBB catalogue fails here by name,
* ``execute_python`` runs through the sandbox seam,
* the loop still completes: RawData structured turn → real ``compute_evidence`` →
  scripted verdict → ``persona_signals``.
"""

from __future__ import annotations

import importlib
from typing import Any

import pytest
from langchain_core.messages import ToolMessage
from pydantic import BaseModel

from muffin_agent.agents.personas_council.council_graph import PERSONA_BUILDERS
from muffin_agent.agents.personas_council.schemas import AnalystSignal

from ._harness import parallel_tool_turn, patch_llm, patch_mcp, patch_sandbox, tool_turn

pytestmark = pytest.mark.asyncio

_SANDBOX_STDOUT = "42.0\n"


def _persona_module(builder: Any):
    return importlib.import_module(builder.__module__)


def _raw_data_name(module: Any) -> str:
    return next(
        v.__name__
        for v in vars(module).values()
        if isinstance(v, type)
        and issubclass(v, BaseModel)
        and v.__name__.endswith("RawData")
    )


def _signal_cls(module: Any) -> type[AnalystSignal]:
    return next(
        v
        for v in vars(module).values()
        if isinstance(v, type)
        and issubclass(v, AnalystSignal)
        and v is not AnalystSignal
    )


@pytest.mark.parametrize(
    ("slug", "builder"), PERSONA_BUILDERS, ids=[s for s, _ in PERSONA_BUILDERS]
)
async def test_persona_collect_data_calls_every_tool(slug, builder, config):
    """The persona's ReAct loop executes its whole tool allowlist end to end."""
    module = _persona_module(builder)
    mcp_tools: list[str] = module._MCP_TOOLS
    assert mcp_tools, f"{slug} declares no _MCP_TOOLS"

    raw_name = _raw_data_name(module)
    signal_cls = _signal_cls(module)
    evidence = module.compute_evidence_node({})["evidence"]
    verdict = signal_cls(
        signal="hold",
        confidence=0.5,
        reasoning=f"{slug} tool-coverage verdict.",
        evidence=evidence,
    )

    script = (
        # Turn 1 — batch-call every MCP tool + execute_python in parallel.
        parallel_tool_turn(
            *[(name, {"symbol": "AAPL"}) for name in mcp_tools],
            ("execute_python", {"code": "print(6 * 7.0)"}),
        ),
        # Turn 2 — emit the response_format structured extraction (defaults).
        tool_turn(raw_name, {}),
        # Turn 3 — render_verdict's direct structured call.
        verdict,
    )

    with (
        patch_mcp(scenario="aapl"),
        patch_sandbox(execute_output=_SANDBOX_STDOUT),
        patch_llm(*script) as cursor,
    ):
        agent = await builder(config)
        result = await agent.ainvoke(
            {"ticker": "AAPL", "as_of_date": "2026-06-09", "query": None},
            config=config,
        )

    # Pipeline completed: one signal from THIS persona, all 3 turns consumed.
    signals = result["persona_signals"]
    assert len(signals) == 1 and signals[0]["agent_id"] == slug
    assert cursor.consumed == 3

    # The 2nd model call saw a ToolMessage per issued call — every tool executed.
    tool_msgs = [m for m in cursor.inputs[1] if isinstance(m, ToolMessage)]
    by_name = {m.name: m for m in tool_msgs}
    expected = set(mcp_tools) | {"execute_python"}
    assert set(by_name) == expected, (
        f"{slug}: tools not executed: {expected - set(by_name)}"
    )

    # Every MCP tool returned its fixture envelope (not an error ToolMessage) —
    # this is what catches a missing fixture or a tool name drifting from the
    # OpenBB catalogue.
    for name in mcp_tools:
        msg = by_name[name]
        assert msg.status != "error", f"{slug}/{name}: tool errored: {msg.content!r}"
        assert '"results"' in msg.content, (
            f"{slug}/{name}: not a fixture envelope: {msg.content[:120]!r}"
        )
    # execute_python ran through the sandbox seam.
    assert by_name["execute_python"].content.strip() == _SANDBOX_STDOUT.strip()
