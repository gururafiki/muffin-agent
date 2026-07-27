# `/threads/{id}/history` latency scales with a thread's namespace count

**Measured 2026-07-27 on the deployed Oracle node.** Supersedes the "flat ~17–27 s
checkpointer read" characterisation in
[`docs/backend-notes/2026-07-23-getstate-latency.md`](2026-07-23-getstate-latency.md)
and in the [subagent-execution-tree design spec](../superpowers/specs/2026-07-23-subagent-execution-tree-design.md).

## Summary

`POST /threads/{id}/history` takes ~1 s **per namespace in the thread**, even when only
the root namespace's history is requested. The database is not involved: the query plan
is a 0.105 ms index scan. The wait is app-side and near-zero-CPU, i.e. serialized
awaits — an N+1 that appears to touch every namespace regardless of the request.

This matters because muffin-ui reconstructs a run's execution tree from this endpoint
(`muffin-ui/src/lib/agent/run-history.ts`). Correctness is fine; interaction cost is not.

## Evidence

Two threads with **near-identical checkpoint counts** and a 6.6× latency difference:

| thread | graph | namespaces | checkpoints | history (limit 10) | payload |
|---|---|---|---|---|---|
| `019f98e1-b104-7742-a893-4b1a9a388366` | criteria_analysis | **27** | 199 | **27.31 s** | 208 KB |
| `019f81a0-0ccd-7301-9710-e4ccea8ddb95` | trading_decision | **7** | 196 | **4.13 s** | 846 KB |

27.31/27 ≈ 1.01 s per namespace; 4.13/7 ≈ 0.59 s. Note the **larger** payload is the
**faster** request — this is not data volume.

### Ruled out by measurement

- **Not the network, Cloudflare or Traefik.** The 27.31 s above was measured from
  *inside* the `langgraph-api` container against `http://localhost:8000`.
- **Not the query.** ```
  EXPLAIN (ANALYZE, BUFFERS) SELECT c.checkpoint, c.metadata, c.checkpoint_ns, c.checkpoint_id
  FROM checkpoints c WHERE c.thread_id = '019f98e1-…' AND c.checkpoint_ns = ''
  ORDER BY c.checkpoint_id DESC LIMIT 10;
  ```
  → `Index Scan Backward using checkpoints_pkey`, `Buffers: shared hit=12`,
  **Execution Time: 0.105 ms**.
- **Not bloat.** `pg_stat_user_tables` reports `n_dead_tup = 0` for `checkpoints`,
  `checkpoint_blobs` and `checkpoint_writes`.
- **Not missing indexes.** `checkpoints_pkey (thread_id, checkpoint_ns, checkpoint_id)`,
  `checkpoints_checkpoint_id_idx (thread_id, checkpoint_id DESC)`,
  `checkpoint_blobs_pkey (thread_id, checkpoint_ns, channel, version)` all present.
- **Not CPU or memory pressure.** During the 27 s: `langgraph-api` 0.59 % CPU /
  670 MiB of 3 GiB; `supabase-db` 0.10 % CPU / 229 MiB of 1.5 GiB. Node load average
  0.23. Postgres connections well under `max_connections = 100`.

Near-zero CPU across 27 seconds means the process is **waiting**, not computing.

## Incidental finding: storage is one bad thread, not general growth

`checkpoint_blobs` is 1878 MB total on a **4.9 MB heap** — essentially all TOAST.

By channel:

| channel | rows | bytes |
|---|---|---|
| `messages` | 15587 | **1791 MB (95 %)** |
| `subagent_runs` | 104 | 350 kB |
| `tool_runs` | 128 | 113 kB |

By thread: **1763 MB of the 1878 MB belongs to a single errored council thread**
(`019f8476-06fd-70bd-97a8-011c7f2bc4d9`, 2026-07-21). Every other thread combined is
under 30 MB.

Two corrections follow:

1. Removing the `agent_capture` channels will **not** meaningfully shrink checkpoints
   (350 kB of 1878 MB). That change is justified by removing observability-as-state,
   not by storage.
2. Pruning that one thread would bring the tables to ~115 MB, which makes "include
   checkpoints in the nightly backup" cheap — currently they are excluded
   (`muffin-deployment/stack/muffin-db-backup.sh`), which loses run history on restore.

## Reproduction

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

Namespace count for a thread:

```sql
SELECT count(DISTINCT checkpoint_ns) FROM checkpoints WHERE thread_id = '<THREAD_ID>';
```

## Suggested upstream report

Not yet filed — it targets `langchain-ai/langgraph` (or LangGraph Platform support,
since `langgraph-api` ships inside the image built by `langgraph-cli` and is not
vendored here). The report should state: *history requests for a single namespace incur
~1 s of app-side latency per OTHER namespace in the thread, with the DB at 0.1 ms and
the process at <1 % CPU; likely a per-namespace round trip in the endpoint's task
resolution that is not scoped to the requested checkpoint.* Include the table above.

Server version at time of measurement: see `langgraph-cli` in
[`pyproject.toml`](../../pyproject.toml); API reachable at `muffin-api.rafiki.guru`.
