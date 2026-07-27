# API read latency is uncached MCP tool discovery, not the checkpointer

**Measured 2026-07-27.** This note replaces two earlier, wrong diagnoses of the same
symptom:

1. `docs/backend-notes/2026-07-23-getstate-latency.md` — "a flat ~17–27 s checkpointer
   read".
2. This file's own first version (`…-history-endpoint-namespace-n-plus-1.md`), which
   claimed "~1 s of app-side latency per namespace in the thread" and drafted an
   upstream `langgraph-api` bug report. **That report was never filed and must not be —
   the bug was ours.**

## Summary

`POST /threads/{id}/history` took 27.3 s on a criteria thread and 4.1 s on a trading
thread. Neither the database nor LangGraph was responsible.

LangGraph Platform rebuilds a **factory-registered** graph on *every* API request
(`langgraph_api/graph.py` → `invoke_factory`, no caching). All five muffin graphs are
factories (`langgraph.json` → `make_graph`), so a plain read rebuilds the whole graph
before touching a checkpoint. Every agent factory in that graph called `get_tools`, and
`get_tools` opened a **fresh MCP session per call** just to list tools — ~1.1 s each.

| graph | MCP round trips per build | measured history read | per round trip |
|---|---|---|---|
| `criteria_analysis` | **23** | 27.31 s | 1.19 s |
| `trading_decision` | **4** | 4.13 s | 1.03 s |
| `council` | 13 | — | — |
| `research` | 3 | — | — |

Two graphs, two round-trip counts, one constant. That is the whole of the latency.

**Fix:** `get_tools` now caches discovered tool lists per MCP connection set
(`agents/data_collection/utils.py`), so a build costs **one** round trip instead of 23,
and zero once the process is warm. Caching is safe because
`MultiServerMCPClient.get_tools()` documents that "a new session will be created for
each tool call": a cached tool holds a connection *spec*, not a live connection.

## Why the earlier diagnoses looked plausible

- **"Per namespace."** The criteria thread happened to have 27 namespaces and 23 MCP
  round trips; the trading thread 7 and 4. Those ratios are close enough that a
  two-point sample cannot separate them. The MCP count is the one that also explains
  `council` and `research`, and the one that survives changing the thread.
- **"The checkpointer is slow."** It never was — see below. The cost is paid *before*
  the first checkpoint row is read, which is why it was flat in `limit`: **limit=1 and
  limit=10 both took ~28 s.** A per-snapshot or per-row cost would have scaled; a
  per-request setup cost does not. That observation was already in the data and should
  have falsified the N+1 theory on the spot.

### Ruled out by measurement

- **Not the network, Cloudflare or Traefik.** 27.31 s was measured from *inside* the
  `langgraph-api` container against `http://localhost:8000`.
- **Not the query.** `EXPLAIN (ANALYZE, BUFFERS)` of the checkpointer's list query:
  `Index Scan Backward using checkpoints_pkey`, `Buffers: shared hit=12`,
  **Execution Time: 0.105 ms**.
- **Not bloat.** `n_dead_tup = 0` on `checkpoints`, `checkpoint_blobs`,
  `checkpoint_writes`.
- **Not missing indexes.** `checkpoints_pkey`, `checkpoints_checkpoint_id_idx`,
  `checkpoint_blobs_pkey` all present.
- **Not CPU or memory.** During the 27 s: `langgraph-api` 0.59 % CPU, `supabase-db`
  0.10 %, node load average 0.23. Near-zero CPU across 27 seconds means **waiting on
  I/O** — the clue that pointed at MCP all along.

## Reproduction

Count the round trips one build costs, with the MCP client stubbed:

```python
calls = {"n": 0}
class Fake:
    def __init__(self, *a, **k): pass
    async def get_tools(self):
        calls["n"] += 1
        return []

with patch("muffin_agent.agents.data_collection.utils.MultiServerMCPClient", Fake):
    await make_graph({"configurable": {}})   # criteria_analysis
print(calls["n"])   # 23 before the cache, 1 after
```

End-to-end against the node:

```bash
CID=$(docker ps --filter name=muffin_langgraph-api -q | head -1)
docker exec "$CID" python -c "
import time, urllib.request, json
body = json.dumps({'limit': 10}).encode()
req = urllib.request.Request('http://localhost:8000/threads/<THREAD_ID>/history',
                             data=body, headers={'Content-Type': 'application/json'})
t0 = time.time(); d = urllib.request.urlopen(req).read()
print(f'{time.time()-t0:.2f}s  {len(d)//1024} KB  {len(json.loads(d))} snapshots')"
```

Expect ~1–2 s on the first call after a restart, well under a second thereafter. The
first request per process still pays one round trip; that is by design.

## Scope of the fix

Not only a read-path win. Every *run* also rebuilt its graph, so a council run paid 13
round trips before its first token. The cache removes that too.

Caveat: the cache is per api process with a 15-minute TTL, so a redeployed `openbb-mcp`
whose tool list changed can go unnoticed for up to 15 minutes by an already-running api
process. Restarting `langgraph-api` clears it immediately.

## Incidental finding: checkpoint storage is one bad thread

`checkpoint_blobs` is 1878 MB total on a 4.9 MB heap — essentially all TOAST.

| channel | rows | bytes |
|---|---|---|
| `messages` | 15587 | **1791 MB (95 %)** |
| `subagent_runs` | 104 | 350 kB |
| `tool_runs` | 128 | 113 kB |

**1763 MB of the 1878 MB belongs to a single errored council thread**
(`019f8476-06fd-70bd-97a8-011c7f2bc4d9`, 2026-07-21). Every other thread combined is
under 30 MB.

Two consequences:

1. Removing the `agent_capture` channels does **not** meaningfully shrink checkpoints
   (350 kB of 1878 MB). That change is justified by removing observability-as-state,
   not by storage.
2. Pruning that one thread brings the tables to ~115 MB, at which point "include
   checkpoints in the nightly backup" is cheap. They are excluded today
   (`muffin-deployment/stack/muffin-db-backup.sh`), which loses run history on restore
   now that checkpoints are the only record of what a run did.
