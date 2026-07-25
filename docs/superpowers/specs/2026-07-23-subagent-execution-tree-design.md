# Recursive sub-agent execution tree

**Date:** 2026-07-23
**Status:** Design — awaiting review
**Scope:** cross-repo — `muffin-agent` (capture + propagation + store offload) and `muffin-ui` (recursive tree UI)

## Problem

Users want to explore the **real execution topology** of a run: every subgraph invocation and every
deepagent `task`-tool subagent as a node, drilling down recursively to any depth, uniformly across
all agents, **both live and on reopened finished runs**. Examples:

- Council: personas/specialists are subagents → click a persona → see *its* subagents
  (`collect_data`, `compute_evidence`, `render_verdict`) → click `collect_data` → see which
  data-collection subagents it called → their tool calls.
- Criteria analysis: click a criterion → see the subagents it executed.
- Same recursion for trading_decision / research / stock_evaluation.

Each node should show **trading-decision-level detail** — the `SubgraphDetail` view today: the Steps
transcript (ReAct loop with inline tool calls) + structured output + tool-runs panel.

### Why today's UI can't do this

The panel is built only from `stream.subgraphsByNode` — the **depth-1 compiled subgraph nodes**
(council personas, trading analysts, criteria stages/workers) — plus registry-seeded stage rows and
captured deep-agent `subagent_runs`. It never surfaces (a) deepagent `task`-tool subagents
(`stream.subagents`, unused), or (b) nesting below depth 1 (a persona's inner `collect_data`).

### Probe finding (measured, decisive)

On a **finished** council thread, `GET /threads/{id}/state?subgraphs=true` returns **0 nested
tasks** (17.7 s) — a completed run has no pending `tasks`, so the nested subgraph states are gone
from the final checkpoint. LangSmith confirms the tree *did* run (persona → `collect_data`;
specialists → fetch nodes). Therefore **the deep tree cannot be reconstructed client-side from a
finished run's final state** — it must be **captured while running**. (A checkpoint-*history* walk
could in theory rebuild it, but every level is a ~17–27 s checkpointer read — the flat latency the
reopen work documented — and reconstruction is complex; rejected.)

This is why `AgentCaptureMiddleware` is the right vehicle: it already runs `aafter_agent` on every
agent (compiled subgraph nodes **and** `task` subagents), and can read its own `checkpoint_ns`.

## Design

Two data tiers feed one recursive UI. Design goal: the tree must be reconstructable at **any depth**
for **both** live and reopened runs, while keeping `thread.values` light so the reopen fix
(hydrate from `thread.values`, ~110 ms) is preserved.

### A. Topology channel — light, rides `thread.values`

A new reducer state channel (`subagent_tree`) accumulates one light record per agent invocation:

```
TreeNode = {
  id: str,             # the checkpoint_ns path (stable, unique per invocation)
  parent_id: str|None, # id whose ns is this node's ns minus the last segment (or None at root)
  name: str,           # agent/subagent name (e.g. "mohnish_pabrai", "collect_data", "web_search")
  kind: "subgraph" | "task",
  status: "ok" | "error" | ...,   # derived from the agent's own tool_runs/messages
  tool_summary: { count: int, tools: [str], ok: int, failed: int, cached: int },
  output_preview: str | None,     # short preview of the agent's structured output, if any
  has_detail: bool,               # whether a heavy detail payload was offloaded (see B)
  started_at: str | None,
}
```

- **Captured** by `AgentCaptureMiddleware`: read `get_config()["configurable"]["checkpoint_ns"]` to
  derive `id`/`parent_id`; summarise the agent's own `tool_runs` into `tool_summary`.
- **Merged up** via a reducer (like `merge_subagent_runs`), so nested captures accumulate on the
  root state.
- **Propagated** through the output-schema boundaries that currently drop such channels — the same
  two-boundary fix `tool_runs` already uses (persona `<Persona>Output`, criteria
  `_CriterionWorkerOutput`, and any other `output_schema`-restricted subgraph — enumerated in the
  plan). The client rebuilds the tree from `parent_id`/`id`.
- **Light** (~tens of nodes × a small record) → `thread.values` stays small → reopen stays ~110 ms.

### B. Detail payloads — heavy, offloaded to the Store, fetched lazily

Per node, the heavy detail (full serialized transcript + tool payloads + full structured output) is
written to the **Store**, mirroring the tool-result-cache pattern:

- Namespace `("subagent_detail", thread_id)`, key `node_id`, value
  `{ messages: [...], tool_runs: [...], output: {...} }`.
- Written best-effort in `aafter_agent` (swallow store errors, like `cache_store`).
- Fetched by the UI **only when a node is expanded**, via `store.getItem(["subagent_detail", threadId], nodeId)`.
- This keeps `thread.values` light **and** delivers the transcript-offload the reopen work deferred
  (Track-2) — the two efforts dovetail. The existing top-level `subagent_runs`/`tool_runs` channels
  can then be slimmed or retired (migration handled in the plan; UI keeps a fallback for old runs).

### C. Recursive UI tree — `muffin-ui`

A new `SubagentTree` component:

- Builds the tree from the `subagent_tree` channel (via `parent_id`/`id`); renders top-level nodes
  as rows (reusing the `SubagentActivity`/`SubAgentRunRow` look).
- Each row expands to a **detail panel = trading-decision-level `SubgraphDetail`** (Steps transcript
  + structured output + `ToolRunsPanel`) **plus that node's own child `SubagentTree`** — recursive,
  to any depth.
- Heavy detail (transcript/tools/output) for an expanded node is fetched lazily from the Store
  (a `useQuery` per expanded node, like the tool-cache provider).
- Because it reads the **captured channel**, it renders identically **live and on history** — no
  dependence on the fragile live-discovery depth.
- Mounted uniformly on every run surface (council, criteria, trading, research, stock_evaluation,
  and the `/calls/[threadId]` detail), replacing/augmenting the current per-agent panels so the
  drill-down behaves the same everywhere.

## Key considerations & risks

- **`checkpoint_ns` fidelity (the crux — validate first).** The tree hinges on each invocation
  having a distinct, prefix-nestable `checkpoint_ns`. Compiled subgraph nodes get
  `<node>:<uuid>`-style namespaces. **Deepagent `task`-tool subagents run via `.ainvoke()` inside
  the task tool** — it must be confirmed that they receive a distinct child `checkpoint_ns` (not the
  parent's) so their tree position is unambiguous. **First implementation step is a spike** that
  instruments the middleware to log `checkpoint_ns` across a real nested run (council persona →
  `collect_data` → data-collection subagent) and confirms the paths nest correctly. If `task`
  subagents don't get a distinct ns, fall back to `SubagentDiscoverySnapshot.parentId`/`depth`
  semantics captured via the task tool's `ls_agent_type` marker + a synthetic id.
- **Propagation enumeration.** The plan must list every `output_schema`-restricted subgraph and add
  the channel: council persona output, criteria `_CriterionWorkerOutput`, the conference subgraphs
  (`InvestmentDebateOutput`/`RiskDebateOutput`), research (`researcher_node` forwards explicitly),
  and confirm the auto-propagating analyst nodes. Missing one = that branch's subtree is invisible.
- **State size.** Only the light tree is in `thread.values`; measure it for a full 19-member council
  and confirm reopen stays ~100 ms.
- **Store growth / GC.** `subagent_detail` entries persist like the tool cache; note a TTL/prune as a
  follow-up (ROADMAP), not in scope here.
- **Backward compatibility.** Old runs have no `subagent_tree` → the UI falls back to today's
  behaviour (`subgraphsByNode` + `subagent_runs`). No migration.
- **`muffin-agent` process.** `main` requires **PR + CodeQL**; every registered graph has an
  enforced integration test — the capture change ships with unit tests for topology capture and
  integration assertions that `subagent_tree` populates for council + criteria + trading.
- **Reopen interplay.** The light-tree + store-offload split is deliberately chosen to preserve the
  just-shipped fast reopen; do not put transcripts in `thread.values`.

## Verification

- **muffin-agent:** `pytest -m unit` for the middleware topology capture (ns → parent/child, tool
  summary); integration tests asserting the `subagent_tree` channel is populated + correctly nested
  for council / criteria_analysis / trading_decision (mock LLM/MCP per the harness); `ruff` + `mypy`.
- **muffin-ui:** `npx tsc --noEmit` + `npx expo export -p web` + a headless drill-down smoke against
  a real thread (expand a council persona → see its `collect_data` child → expand → see it made 0
  tool calls), zero Reanimated errors, screenshot.

## Documentation updates (final step)

- `muffin-agent/CLAUDE.md` — the `agent_capture` section (new `subagent_tree` channel + the
  propagation requirement per output-schema-restricted subgraph).
- `muffin-ui/CLAUDE.md` — the renderers/subagents-panel section (recursive `SubagentTree`, store
  detail fetch, live-vs-history now unified via the captured channel).
- ROADMAP (both repos): the feature + the store-GC follow-up; close the reopen Track-2
  transcript-offload item as superseded.
- Update the `muffin-ui-subagents-panel-discovery` project memory (panel now reflects real captured
  topology, not registry-seeded rows).

## Open questions

- **Channel granularity:** does every agent invocation become a node, or only "meaningful" ones
  (skip pure middleware/util nodes)? Proposal: capture at the `AgentCaptureMiddleware` boundary only
  (real agents), which already excludes framework plumbing — verify this yields the expected
  ~persona/collect_data/data-collection nodes, not the 500 raw LangSmith runs.
- **Node identity across reducer merge:** `checkpoint_ns` is unique per invocation, so the reducer is
  a plain dict-merge keyed by `id`; confirm no collisions under parallel fan-out (Send workers).
