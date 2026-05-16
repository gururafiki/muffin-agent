---
name: research-task-comparison
description: >
  Comparison task type: identify dimensions of comparison first,
  then gather parallel data per entity to support a comparison table.
metadata:
  task_type: comparison
---

# Task Type — Comparison

The writer will produce a **markdown comparison table** plus
per-entity narrative.  Your evidence must let the writer fill every
cell of the table.

## Step 1: Identify the entities and dimensions

Before searching, list:
- **Entities** to compare (typically 2-4).
- **Dimensions** that matter for the user's intent.  Examples:

| Query                            | Likely dimensions                          |
|----------------------------------|--------------------------------------------|
| "Postgres vs MySQL"              | performance, replication, JSON, license    |
| "iPhone 16 vs Pixel 9"           | price, camera, battery, software lifespan  |
| "Python vs Go for microservices" | concurrency, perf, ecosystem, deployment   |

Pick 4-7 dimensions that meaningfully discriminate.

## Step 2: Gather parallel data

For each (entity, dimension) cell, gather one piece of evidence.
Prefer **the same source type per dimension across entities** (e.g.
benchmark results from the same suite, prices from the same retailer)
so the writer's table is apples-to-apples.

## Step 3: Capture trade-offs

For each entity, find at least one piece of evidence about a known
**weakness or trade-off**.  Comparison tables that show only
strengths are uninformative.

## Source priorities

1. **Official specs / docs** for the canonical numbers.
2. **Independent benchmarks** with stated methodology.
3. **User community signal** for real-world adoption notes.

## Evidence chunks

- Per chunk, include the entity it relates to in ``title`` or
  ``snippet`` so the rerank step doesn't merge cells from different
  entities just because URLs look similar.
- It's OK (encouraged) to have 2× #entities × #dimensions chunks —
  this is a high-evidence-density task.

## Common pitfalls

- Picking dimensions the user didn't care about and ignoring ones
  they did.  Re-read the query.
- Asymmetric data — covering entity A on 6 dimensions and entity B
  on 2.  Strive for parity.
- One-sided framing — every entity has trade-offs.
