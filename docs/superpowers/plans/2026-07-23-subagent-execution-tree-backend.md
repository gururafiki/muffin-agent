# Sub-agent execution tree — Phase 1 (backend capture) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Capture a run's execution topology (compiled subgraph nodes + deepagent `task` subagents) into a light `subagent_tree` state channel, and offload each node's heavy detail (transcript + tools + output) to the Store — so the UI (Phase 2, separate plan) can render a recursive drill-down tree that works identically live and on reopened runs.

**Architecture:** Extend `AgentCaptureMiddleware` (which already runs `aafter_agent` on every agent) to (a) read its `checkpoint_ns` and emit one light `TreeNode` into a reducer-merged `subagent_tree` channel, and (b) best-effort write its heavy detail to the Store. Add `subagent_tree` to the `output_schema`-restricted subgraph boundaries that would otherwise drop it (same fix `tool_runs` already has). No graph re-architecture.

**Tech Stack:** Python 3.13, LangGraph/LangChain, `deepagents`, Pydantic/TypedDict state, `langgraph.store.base.BaseStore`, pytest (`-m unit` / `-m integration` with the `tests/integration/_harness`).

## Global Constraints

- **Spec:** `docs/superpowers/specs/2026-07-23-subagent-execution-tree-design.md`. This plan is **Phase 1 (backend) only**; the muffin-ui recursive tree is a separate plan written after this channel lands.
- **This repo is `muffin-agent`; `main` requires a PR + CodeQL.** Work on branch `subagent-execution-tree` (already created).
- **Ruff** (Google-style docstrings, `D401`), **Pydantic** for models, **mypy** clean on `src/`. `pytest` must stay green.
- **Every registered graph keeps its enforced integration test.** The capture change ships with unit tests + integration assertions that `subagent_tree` populates and nests.
- **Keep the channel LIGHT** (rides `thread.values`, which the reopen fix hydrates in ~110 ms) — heavy detail goes to the Store, never the state channel.
- **Capture is unconditional** (no runtime config gate — same rule as the existing `subagent_runs`/`tool_runs`; graphs opt in by declaring the channel; parents that don't declare it drop the records for free).

---

## File Structure

- **Create** `src/muffin_agent/middlewares/agent_capture/tree.py` — the `TreeNode` shape, the `merge_subagent_tree` reducer, `checkpoint_ns` parsing (`ns → id/parent_id/depth`), and `build_tree_node(...)` (summarise the agent's own `tool_runs` + output into a light node). One responsibility: topology record construction.
- **Create** `src/muffin_agent/middlewares/agent_capture/detail_store.py` — `offload_subagent_detail(store, thread_id, node_id, payload)` mirroring `tool_result_cache/cache.py:cache_store` (best-effort `store.aput`, swallow errors). One responsibility: heavy-detail offload.
- **Modify** `src/muffin_agent/middlewares/agent_capture/middleware.py` — declare `subagent_tree` on `AgentCaptureState`; in `_capture`/`aafter_agent` emit the node + offload detail.
- **Modify** `src/muffin_agent/middlewares/agent_capture/__init__.py` — export the new public names.
- **Modify** the `output_schema`-restricted boundaries to carry `subagent_tree` (see Task 5): `agents/personas_council/schemas.py` (+ persona `<Persona>Output`), `agents/criteria_analysis/criterion_evaluation_node.py` (`_CriterionWorkerOutput`), `agents/criteria_analysis/state.py`, the conference `output_schema`s in `agents/trading_decision/graph.py`, and `agents/research/state.py`.
- **Create** `tests/middlewares/test_agent_capture_tree.py` — unit tests (ns parsing, reducer, node build).
- **Modify** the relevant integration tests under `tests/integration/` to assert `subagent_tree` populates + nests.

---

## Task 1: Spike — validate `checkpoint_ns` fidelity — ✅ DONE (validated by controller, 2026-07-23)

Ran the real council + criteria integration graphs with a temporary `print` of
`get_config()["configurable"]["checkpoint_ns"]` inside `AgentCaptureMiddleware._capture`
(env-gated, then reverted — `git diff` clean). **Findings — the ns model is validated, with two
refinements Task 2 must honour:**

1. **Compiled subgraph nesting works and fully encodes the tree.** Real namespaces are pipe-joined
   `<node>:<uuid>` segments, e.g.:
   - council: `mohnish_pabrai:<uuid>|collect_data:<uuid>|AgentCaptureMiddleware.after_agent:<uuid>`
   - criteria: `criterion_evaluation:<uuid>|evaluate:<uuid>|AgentCaptureMiddleware.after_agent:<uuid>`;
     stages like `ticker_classification:<uuid>|AgentCaptureMiddleware.after_agent:<uuid>`.
   So the persona → `collect_data`, and criterion-worker → `evaluate` hierarchy is directly readable.
2. **Refinement A — strip the trailing middleware segment.** The LAST segment is always the
   capturing middleware's own node (`AgentCaptureMiddleware.after_agent:<uuid>`). `node_ids_from_ns`
   MUST drop trailing `AgentCaptureMiddleware.*:<uuid>` segment(s) before computing `id`/`parent_id`.
3. **Refinement B — reconstruct intermediate nodes from ns segments.** Only the leaf agents capture
   (e.g. `collect_data`); the intermediate level (`mohnish_pabrai` the persona subgraph) has no
   capture of its own and appears only as an ancestor ns prefix. So the tree is built by splitting
   each captured node's cleaned ns into `<name>:<uuid>` segments — each segment is a node
   (`name` = the segment's node name, `id` = the cumulative prefix), and captured detail attaches to
   the deepest segment. The backend emits ONE record per capture keyed by its cleaned ns; ancestor
   structural nodes are reconstructed by the consumer (Phase-2 UI) from the segment prefixes. This
   keeps the channel minimal.

**Still open (minor, fallback covers it):** the mocked integration tests never invoke the deepagent
`task` subagents (`subagent=False` throughout — the scripted model returns structured output without
calling the data-collection subagents), so the `task`-subagent ns is unconfirmed. Deepagent `task`
subagents run via `subagent.ainvoke(...)` with ambient config; they most likely receive a further
nested ns segment, but if they instead share the parent's ns, the `_running_as_subagent()` marker +
a minted `|task:<uuid>` id (already in Task 2's fallback) handles it. Confirm opportunistically when
a real `task` call is exercised; it does not block Phase 1.

---

## Task 2: `TreeNode` shape, reducer, and namespace parsing

**Files:**
- Create: `src/muffin_agent/middlewares/agent_capture/tree.py`
- Test: `tests/middlewares/test_agent_capture_tree.py`

**Interfaces:**
- Produces: `TreeNode` (TypedDict), `merge_subagent_tree(left, right) -> dict[str, TreeNode]`, `node_ids_from_ns(checkpoint_ns: str | None) -> tuple[str, str | None]` (returns `(id, parent_id)`), `build_tree_node(*, node_id, parent_id, name, kind, tool_runs, output) -> TreeNode`.

- [ ] **Step 1: Write failing unit tests**

```python
# tests/middlewares/test_agent_capture_tree.py
import pytest
from muffin_agent.middlewares.agent_capture.tree import (
    node_ids_from_ns, build_tree_node, merge_subagent_tree,
)

def test_ns_parsing_root():
    assert node_ids_from_ns("") == ("__root__", None)
    assert node_ids_from_ns(None) == ("__root__", None)

def test_ns_parsing_nested():
    # LangGraph ns segments are pipe-separated "<node>:<task_id>"
    assert node_ids_from_ns("persona:abc") == ("persona:abc", "__root__")
    assert node_ids_from_ns("persona:abc|collect:def") == (
        "persona:abc|collect:def", "persona:abc",
    )

def test_ns_parsing_strips_trailing_middleware_segment():
    # Real namespaces (Task-1 spike): the capturing middleware's own node is the
    # trailing segment and must be stripped before deriving id/parent.
    assert node_ids_from_ns(
        "mohnish_pabrai:a|collect_data:b|AgentCaptureMiddleware.after_agent:c"
    ) == ("mohnish_pabrai:a|collect_data:b", "mohnish_pabrai:a")
    assert node_ids_from_ns(
        "ticker_classification:a|AgentCaptureMiddleware.after_agent:b"
    ) == ("ticker_classification:a", "__root__")
    assert node_ids_from_ns("AgentCaptureMiddleware.after_agent:a") == ("__root__", None)

def test_build_node_summarises_tools():
    runs = [
        {"tool": "news_company", "status": "ok"},
        {"tool": "news_company", "status": "error"},
        {"tool": "equity_price", "status": "ok", "cache_hit": True},
    ]
    n = build_tree_node(node_id="p:1", parent_id="__root__", name="pabrai",
                        kind="subgraph", tool_runs=runs, output={"signal": "hold"})
    assert n["name"] == "pabrai" and n["parent_id"] == "__root__"
    assert n["tool_summary"] == {"count": 3, "tools": ["news_company", "equity_price"],
                                 "ok": 2, "failed": 1, "cached": 1}
    assert n["output_preview"] and n["has_detail"] is True

def test_reducer_merges_by_id():
    a = {"p:1": {"id": "p:1"}}; b = {"p:1|c:2": {"id": "p:1|c:2"}}
    assert set(merge_subagent_tree(a, b)) == {"p:1", "p:1|c:2"}
```

Run: `pytest tests/middlewares/test_agent_capture_tree.py -v` → FAIL (module missing).

- [ ] **Step 2: Implement `tree.py`**

```python
"""Execution-topology records for the sub-agent tree (light; rides thread.values)."""
from __future__ import annotations
from typing import Any, Literal, TypedDict

_ROOT = "__root__"

class ToolSummary(TypedDict):
    count: int
    tools: list[str]
    ok: int
    failed: int
    cached: int

class TreeNode(TypedDict):
    id: str
    parent_id: str | None
    name: str
    kind: Literal["subgraph", "task"]
    status: Literal["ok", "error"]
    tool_summary: ToolSummary
    output_preview: str | None
    has_detail: bool

def _strip_capture_segments(segments: list[str]) -> list[str]:
    """Drop trailing ``AgentCaptureMiddleware.*:<uuid>`` segments — the capturing
    middleware's OWN node is always the last ns segment (validated in Task 1)."""
    while segments and segments[-1].split(":", 1)[0].startswith("AgentCaptureMiddleware"):
        segments = segments[:-1]
    return segments

def node_ids_from_ns(checkpoint_ns: str | None) -> tuple[str, str | None]:
    """Map a LangGraph ``checkpoint_ns`` to ``(id, parent_id)``.

    Namespaces are ``|``-joined ``<node>:<task_id>`` segments; the LAST segment is
    the capturing middleware's own node and is stripped. The id is then the
    cleaned ns; the parent is the cleaned ns minus its last segment
    (``__root__`` at depth 1). Ancestor structural nodes (levels that never
    capture, e.g. the persona subgraph) are reconstructed by the consumer from
    the ``<name>:<uuid>`` segment prefixes.
    """
    if not checkpoint_ns:
        return _ROOT, None
    segments = _strip_capture_segments(checkpoint_ns.split("|"))
    if not segments:
        return _ROOT, None
    node_id = "|".join(segments)
    if len(segments) == 1:
        return node_id, _ROOT
    return node_id, "|".join(segments[:-1])

def _summarise(tool_runs: list[dict[str, Any]]) -> ToolSummary:
    tools: list[str] = []
    ok = failed = cached = 0
    for r in tool_runs:
        name = r.get("tool")
        if name and name not in tools:
            tools.append(name)
        if r.get("status") == "error":
            failed += 1
        else:
            ok += 1
        if r.get("cache_hit"):
            cached += 1
    return {"count": len(tool_runs), "tools": tools, "ok": ok, "failed": failed, "cached": cached}

def build_tree_node(*, node_id: str, parent_id: str | None, name: str,
                    kind: Literal["subgraph", "task"], tool_runs: list[dict[str, Any]],
                    output: Any) -> TreeNode:
    """Build a light topology node summarising what this agent did."""
    summary = _summarise(tool_runs or [])
    preview = None
    if output:
        preview = (str(output)[:280]) or None
    status: Literal["ok", "error"] = "error" if summary["failed"] and not summary["ok"] else "ok"
    return {
        "id": node_id, "parent_id": parent_id, "name": name, "kind": kind,
        "status": status, "tool_summary": summary, "output_preview": preview,
        "has_detail": True,
    }

def merge_subagent_tree(left: dict[str, TreeNode] | None,
                        right: dict[str, TreeNode] | None) -> dict[str, TreeNode]:
    """Reducer: accumulate tree nodes across nested/parallel captures (keyed by id)."""
    return {**(left or {}), **(right or {})}
```

Run: `pytest tests/middlewares/test_agent_capture_tree.py -v` → PASS.

- [ ] **Step 3: `ruff` + `mypy`, then commit**

```bash
ruff check src/muffin_agent/middlewares/agent_capture/tree.py tests/middlewares/test_agent_capture_tree.py
ruff format src/muffin_agent/middlewares/agent_capture/tree.py tests/middlewares/test_agent_capture_tree.py
mypy src/muffin_agent/middlewares/agent_capture/tree.py
git add src/muffin_agent/middlewares/agent_capture/tree.py tests/middlewares/test_agent_capture_tree.py
git commit -m "feat(capture): TreeNode shape + ns parsing + reducer for the sub-agent tree"
```
*(If the Task-1 spike chose the `task`-uuid fallback, add a `mint_task_id(parent_id)` helper here + a test for it, and use it in Task 3.)*

---

## Task 3: Emit the tree node from `AgentCaptureMiddleware`

**Files:**
- Modify: `src/muffin_agent/middlewares/agent_capture/middleware.py`
- Modify: `src/muffin_agent/middlewares/agent_capture/__init__.py`
- Test: extend `tests/middlewares/test_agent_capture_tree.py`

**Interfaces:**
- Consumes: `tree.py` (Task 2), the middleware's existing `build_tool_records`, `get_config`.
- Produces: `subagent_tree` channel on `AgentCaptureState`.

- [ ] **Step 1: Declare the channel + emit the node**

In `middleware.py`: add to `AgentCaptureState`:
```python
subagent_tree: NotRequired[Annotated[dict[str, Any], merge_subagent_tree]]
```
(import `merge_subagent_tree`, `node_ids_from_ns`, `build_tree_node` from `.tree`.)

In `_capture(...)`, after building `records = build_tool_records(...)`, read the ns and emit the node:
```python
try:
    cfg = get_config()
    ns = cfg.get("configurable", {}).get("checkpoint_ns") if isinstance(cfg, dict) else None
except Exception:
    ns = None
node_id, parent_id = node_ids_from_ns(ns)
kind = "task" if _running_as_subagent() else "subgraph"
output = state.get("structured_response")
updates["subagent_tree"] = {
    node_id: build_tree_node(
        node_id=node_id, parent_id=parent_id, name=self._name, kind=kind,
        tool_runs=records or [], output=output,
    )
}
```
Add `subagent_tree` to `AgentCaptureParentMiddleware`'s declared channels too (so a parent that only declares the parent middleware still collects merged-up nodes).

- [ ] **Step 2: Unit test the emission with a stubbed config**

Add a test that instantiates `AgentCaptureMiddleware(name="pabrai")`, monkeypatches `get_config` to return `{"configurable": {"checkpoint_ns": "pabrai:1"}}`, calls `_capture({"messages": [...with one tool call...]})`, and asserts `updates["subagent_tree"]["pabrai:1"]["name"] == "pabrai"` and the tool_summary is populated.

Run: `pytest tests/middlewares/test_agent_capture_tree.py -v` → PASS.

- [ ] **Step 3: `ruff`/`mypy`/commit**

```bash
ruff check src/muffin_agent/middlewares/agent_capture/ tests/middlewares/test_agent_capture_tree.py
mypy src/muffin_agent/middlewares/agent_capture/
git add -A && git commit -m "feat(capture): emit subagent_tree nodes from AgentCaptureMiddleware"
```

---

## Task 4: Offload heavy per-node detail to the Store

**Files:**
- Create: `src/muffin_agent/middlewares/agent_capture/detail_store.py`
- Modify: `src/muffin_agent/middlewares/agent_capture/middleware.py`
- Test: `tests/middlewares/test_agent_capture_detail_store.py`

**Interfaces:**
- Consumes: `runtime.store` (available in `aafter_agent`), the existing `serialize_messages`.
- Produces: `offload_subagent_detail(store, thread_id, node_id, *, messages, tool_runs, output) -> bool`; Store layout `("subagent_detail", thread_id)` key `node_id`.

- [ ] **Step 1: Write failing test (in-memory store)**

```python
# tests/middlewares/test_agent_capture_detail_store.py
import pytest
from langgraph.store.memory import InMemoryStore
from muffin_agent.middlewares.agent_capture.detail_store import offload_subagent_detail

@pytest.mark.asyncio
async def test_offload_roundtrip():
    store = InMemoryStore()
    ok = await offload_subagent_detail(store, "t1", "p:1",
        messages=[{"type": "human", "content": "hi"}], tool_runs=[{"tool": "x"}], output={"s": 1})
    assert ok
    item = await store.aget(("subagent_detail", "t1"), "p:1")
    assert item.value["messages"] and item.value["tool_runs"] and item.value["output"] == {"s": 1}

@pytest.mark.asyncio
async def test_offload_none_store_is_noop():
    assert await offload_subagent_detail(None, "t1", "p:1", messages=[], tool_runs=[], output=None) is False
```

Run: FAIL (module missing).

- [ ] **Step 2: Implement `detail_store.py`** (mirror `tool_result_cache/cache.py:cache_store`)

```python
"""Offload heavy per-subagent detail (transcript + tools + output) to the Store.

Layout mirrors the tool-result cache: namespace ``("subagent_detail", thread_id)``,
key = the tree node id, value = ``{messages, tool_runs, output}``. Best-effort —
store failures are swallowed so capture never breaks a run.
"""
from __future__ import annotations
import logging
from typing import Any
from langgraph.store.base import BaseStore

logger = logging.getLogger(__name__)

async def offload_subagent_detail(
    store: BaseStore | None, thread_id: str, node_id: str, *,
    messages: list[dict[str, Any]], tool_runs: list[dict[str, Any]], output: Any,
) -> bool:
    if store is None:
        return False
    try:
        await store.aput(("subagent_detail", thread_id), node_id,
                         {"messages": messages, "tool_runs": tool_runs, "output": output})
    except Exception:
        logger.debug("offload_subagent_detail failed for %s/%s", thread_id, node_id, exc_info=True)
        return False
    return True
```

- [ ] **Step 3: Call it from `aafter_agent`**

In `middleware.py` `aafter_agent` (async only — sync `after_agent` skips the offload), after `_capture`, resolve `thread_id` from `get_config()["configurable"]["thread_id"]` and the store from `runtime.store`, and offload the heavy detail for the node just built:
```python
await offload_subagent_detail(
    runtime.store, thread_id, node_id,
    messages=serialize_messages(messages), tool_runs=records or [], output=output,
)
```
(Share `node_id`/`records`/`output`/`serialize_messages(messages)` with `_capture` — refactor `_capture` to return the node id + payload, or recompute; keep it DRY.)

Run: `pytest tests/middlewares/test_agent_capture_detail_store.py -v` → PASS.

- [ ] **Step 4: `ruff`/`mypy`/commit**

```bash
git add -A && git commit -m "feat(capture): offload heavy subagent detail to the Store (light tree stays in state)"
```

---

## Task 5: Propagate `subagent_tree` through the output-schema boundaries

**Files (one channel add per boundary, mirroring the existing `tool_runs`):**
- Modify: `src/muffin_agent/agents/personas_council/schemas.py` (+ each `<Persona>Output` / `PersonaState`)
- Modify: `src/muffin_agent/agents/criteria_analysis/criterion_evaluation_node.py` (`_CriterionWorkerOutput`, line ~71) + `agents/criteria_analysis/state.py`
- Modify: `src/muffin_agent/agents/trading_decision/graph.py` (`InvestmentDebateOutput` / `RiskDebateOutput` if they must carry it) + `TradingDecisionState`
- Modify: `src/muffin_agent/agents/research/state.py` (declare channel; `researcher_node` forwards it explicitly like it does `tool_runs`)

**Interfaces:** each parent state that should collect nodes declares `subagent_tree: NotRequired[Annotated[dict[str, Any], merge_subagent_tree]]`; each `output_schema`-restricted subgraph includes `subagent_tree` in its output type (else its subtree is dropped at the boundary).

- [ ] **Step 1: Add the channel everywhere `tool_runs` already crosses a boundary**

For each file above, replicate the exact `tool_runs` treatment for `subagent_tree` (same reducer `merge_subagent_tree`, re-exported from `personas_council/schemas.py` like `merge_tool_runs` is). E.g. in `criterion_evaluation_node.py`:
```python
subagent_tree: Annotated[dict[str, Any], merge_subagent_tree]   # in _CriterionWorkerOutput + the worker State
```
and the `package` node forwards `state.get("subagent_tree")` alongside `tool_runs` if it re-homes them.

- [ ] **Step 2: Verify propagation via the existing compile/output-schema tests**

Council has `tests/agents/test_personas_council/personas/test_all_personas.py::test_persona_subgraph_compiles` asserting `output_schema == {persona_signals, tool_runs}`. Update it to `{persona_signals, tool_runs, subagent_tree}` and run:
```bash
pytest tests/agents/test_personas_council/personas/test_all_personas.py -v
```
Expected: PASS (proves each persona now propagates the channel). Do the analogous assertion for the criteria worker + conference outputs.

- [ ] **Step 3: `ruff`/`mypy`/commit**

```bash
git add -A && git commit -m "feat(capture): propagate subagent_tree through persona/criteria/conference/research output schemas"
```

---

## Task 6: Integration assertions — the tree populates + nests end-to-end

**Files:**
- Modify: `tests/integration/test_council_graph_e2e.py`, `tests/integration/test_criteria_analysis.py` (or the closest existing e2e per `COVERED_GRAPHS`)

- [ ] **Step 1: Assert `subagent_tree` on a real (mocked-boundary) run**

In the council e2e (which already runs the real compiled graph with `patch_llm`/`patch_mcp`), after invoking, assert:
```python
tree = result.get("subagent_tree") or {}
assert tree, "subagent_tree should be populated"
# at least one persona node at depth 1 and (if the scripted model calls a tool/subagent) a nested child
names = {n["name"] for n in tree.values()}
assert names & {"warren_buffett", "mohnish_pabrai"}   # persona nodes present
parents = {n["parent_id"] for n in tree.values()}
assert "__root__" in parents  # depth-1 nodes rooted correctly
```
Script the mocked persona model to make ≥1 tool call in one persona so a nested node + non-empty `tool_summary` is asserted.

- [ ] **Step 2: Run the integration suite for the touched graphs**

Run: `pytest -m integration tests/integration/test_council_graph_e2e.py tests/integration/test_criteria_analysis.py -v`
Expected: PASS.

- [ ] **Step 3: Full gate + commit**

```bash
ruff check src/ tests/ && mypy src/ && pytest -q
git add -A && git commit -m "test(capture): assert subagent_tree populates + nests for council + criteria"
```

---

## Task 7: Docs

**Files:** `CLAUDE.md`, `ROADMAP.md`

- [ ] **Step 1: Update `CLAUDE.md`**

In the `agent_capture/` bullet, document the new `subagent_tree` channel (light topology: `checkpoint_ns → id/parent_id`, `tool_summary`, `has_detail`), the Store offload layout `("subagent_detail", thread_id)/node_id`, and the propagation requirement (every `output_schema`-restricted subgraph must declare `subagent_tree` like `tool_runs`, else its subtree is dropped).

- [ ] **Step 2: ROADMAP**

Add the Phase-1 entry (done) + the Phase-2 pointer (muffin-ui recursive tree) + a store-GC follow-up for `subagent_detail`. Note this supersedes the reopen Track-2 transcript-offload.

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md ROADMAP.md && git commit -m "docs(capture): document subagent_tree channel + detail store + propagation rule"
```

---

## Self-Review

**Spec coverage (Phase 1):** topology channel (Tasks 2–3) ✓; store offload (Task 4) ✓; propagation across boundaries (Task 5) ✓; `checkpoint_ns` risk spiked first (Task 1) ✓; integration proof (Task 6) ✓; docs (Task 7) ✓. Frontend (Tier C) is explicitly the separate Phase-2 plan.

**Placeholder scan:** all code steps carry concrete code; the only branch point is Task-1's spike decision, which Task 2 Step 3 handles explicitly (primary ns id vs task-uuid fallback).

**Type consistency:** `node_ids_from_ns` → `(id, parent_id)` used identically in `tree.py` tests and `middleware.py` (Task 3). `TreeNode` fields (`tool_summary`, `output_preview`, `has_detail`) match across `build_tree_node`, the reducer, and the integration assertions. `offload_subagent_detail` signature is identical in its test and the `aafter_agent` call site.

**Open gate:** Task 1's `checkpoint_ns` finding may flip the `id` derivation for `task` subagents (documented fallback); it does not change the channel shape, the store layout, or the propagation edits.
