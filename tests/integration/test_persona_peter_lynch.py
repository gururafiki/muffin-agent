"""E2E integration test — Peter Lynch persona subgraph.

Worked example #2. This drives a persona's **full** three-node subgraph end to
end — the first test to do so:

    collect_data (real ReAct)  →  compute_evidence (real)  →  render_verdict (LLM)

The existing persona unit tests exercise each node in isolation (they never run
``collect_data``'s ReAct loop, because the old ``FakeLLM`` can't). The
``ScriptedChatModel`` seam closes that gap with one shared timeline across all
three nodes:

1. ``collect_data`` turn 1 — call an MCP tool (fixture-backed).
2. ``collect_data`` turn 2 — emit the ``response_format`` schema tool
   (``PeterLynchRawData``); ``create_agent`` parses it into ``structured_response``
   and the auto-unpack middleware writes its fields into state.
3. ``render_verdict`` — the direct ``get_chat_model_for_role(schema=...)`` call
   returns the scripted ``PeterLynchSignal``.

``compute_evidence`` runs **for real** in between (note #3 — never mock what is
deterministic). We prove it ran on the genuinely-unpacked raw data by inspecting
the verdict prompt: it renders ``Revenue CAGR:`` only when the computed
``evidence.growth.revenue_cagr`` is non-null, which is only possible if the
scripted ``PeterLynchRawData`` flowed collect_data → state → compute_evidence.

⚠️  KNOWN-BUG XFAIL — see ``docs/integration-testing.md`` ("Bug surfaced by this
suite") and ``roadmap.md``. Authoring this example exposed a systemic,
pre-existing break in compiled-subagent composition that makes the council, all
three trading-decision graphs, and the persona CLI non-functional end to end
(masked because every graph-level test stubs these subagents). Two layers:

* **Input mapping** — ``add_node(..., input_schema=agent.input_schema)`` passes a
  property-less ``RootModel`` (every ``.with_state_schema(...)`` agent yields one),
  so LangGraph maps ``{}`` into the node and ``_coerce_state`` raises before any
  model call.
* **Output propagation** — the sub-agent's structured fields are
  ``OmitFromSchema(input=True, output=True)``, so the unpacked values are stripped
  from its output and never reach ``compute_evidence``.

This test pins the *correct* end-to-end behavior, so it will turn into a loud
``XPASS`` (strict) the day the architecture is fixed — at which point the marker
should be removed.
"""

from __future__ import annotations

import pytest

from muffin_agent.agents.personas_council.personas.peter_lynch import (
    PeterLynchSignal,
    build_peter_lynch_agent,
    compute_evidence_node,
)

from ._harness import patch_llm, patch_mcp, patch_sandbox, tool_turn

pytestmark = pytest.mark.asyncio


# A realistic "raw data" extraction the scripted collect_data step emits as its
# response_format payload (oldest → newest series, AAPL-flavoured).
RAW_DATA = {
    "revenue_series": [383285.0, 391035.0, 410245.0],
    "eps_series": [6.13, 6.08, 6.62],
    "free_cash_flow_series": [99584.0, 108807.0, 102340.0],
    "debt_to_equity_latest": 0.45,
    "operating_margin_latest": 0.31,
    "pe_ratio_latest": 30.0,
    "insider_trades": [
        {"transaction_shares": 25000},
        {"transaction_shares": -8000},
    ],
    "company_news": [
        {"sentiment": "positive"},
        {"sentiment": "positive"},
        {"sentiment": "negative"},
    ],
    "market_cap": 3.02e12,
}


def _scripted_verdict() -> PeterLynchSignal:
    """Build the verdict the LLM would emit — reusing the real evidence math."""
    evidence = compute_evidence_node(RAW_DATA)["evidence"]
    return PeterLynchSignal(
        signal="buy",
        confidence=0.66,
        reasoning="GARP-ish: steady growth, keep an eye on the PEG.",
        evidence=evidence,
    )


async def test_compute_evidence_is_deterministic_real_math():
    """The compute node is never mocked — it derives evidence from raw data.

    This layer works today; only the *subgraph composition* is broken (see the
    module docstring + the xfail below).
    """
    evidence = compute_evidence_node(RAW_DATA)["evidence"]
    # Revenue CAGR over the 3-point series ≈ 3.5% — a real, reproducible number.
    assert evidence.growth.revenue_cagr == pytest.approx(0.0346, abs=5e-4)
    assert evidence.fundamentals.debt_to_equity == 0.45
    assert evidence.valuation.peg_ratio is not None  # P/E ÷ EPS-CAGR computed
    assert 0 <= evidence.weighted_score <= 10


async def test_full_subgraph_runs_collect_compute_verdict(config):
    """Drive collect_data → compute_evidence → render_verdict end to end."""
    script = (
        tool_turn("equity_fundamental_metrics", {"symbol": "AAPL", "period": "annual"}),
        tool_turn("PeterLynchRawData", RAW_DATA),
        _scripted_verdict(),
    )
    with patch_mcp(scenario="aapl"), patch_sandbox(), patch_llm(*script) as cursor:
        agent = await build_peter_lynch_agent(config)
        result = await agent.ainvoke(
            {"ticker": "AAPL", "as_of_date": "2026-06-09", "query": None},
            config=config,
        )

    # The subgraph emits exactly one persona signal (its public output contract).
    signals = result["persona_signals"]
    assert len(signals) == 1
    assert signals[0]["agent_id"] == "peter_lynch"
    assert signals[0]["signal"] == "buy"
    assert signals[0]["confidence"] == pytest.approx(0.66)

    # All three model turns were consumed: 1 MCP call + 1 response_format turn
    # (collect_data) + 1 verdict turn (render_verdict).
    assert cursor.consumed == 3

    # subagent_tree: the collect_data ReAct agent is the sole capturing node
    # here (compute_evidence/render_verdict are plain/LLM-direct, not agents),
    # and it runs at the graph root (this test drives the persona subgraph
    # directly, not nested inside the council) — so it roots at "__root__"
    # and its tool_summary reflects the ONE real (fixture-backed) MCP tool
    # call scripted above, proving tool_runs really feeds the tree.
    tree = result.get("subagent_tree") or {}
    assert tree, "subagent_tree should be populated"
    [node] = list(tree.values())
    assert node["name"] == "peter_lynch_data_collection"
    assert node["parent_id"] == "__root__"
    assert node["tool_summary"] == {
        "count": 1,
        "tools": ["equity_fundamental_metrics"],
        "ok": 1,
        "failed": 0,
        "cached": 0,
    }

    # Proof the REAL compute_evidence ran on the genuinely-unpacked raw data:
    # the verdict prompt renders "Revenue CAGR:" only when revenue_cagr is
    # non-null, which requires PeterLynchRawData → state → compute_evidence.
    verdict_prompt = cursor.last_system_prompt()
    assert "Peter Lynch" in verdict_prompt
    assert "Revenue CAGR:" in verdict_prompt
