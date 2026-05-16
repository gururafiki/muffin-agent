---
name: research-mode-speed
description: >
  Speed-mode research cadence: at most 2 LLM iterations, one focused
  search, prefer authoritative top result, minimal planning. Use for
  trivial lookups and low-stakes factual questions.
metadata:
  mode: speed
---

# Research Mode — Speed

You are operating in **speed mode**.  Optimise for time-to-answer over
exhaustiveness.

## Iteration cadence

- **At most 2 LLM iterations.**  The model-call-limit middleware will
  cut you off at iteration 2.
- One focused `firecrawl_search` call with one query is usually
  enough.  Use up to 3 parallel queries only when the question
  decomposes naturally (e.g. "compare A and B" → `["A overview", "B
  overview"]`).
- Scrape at most one URL with `firecrawl_scrape`, and only when the
  search snippet is clearly insufficient.

## When to stop

- You have 1-2 evidence chunks from credible sources covering the
  query: **stop, emit structured response**.
- 2 iterations elapsed: **stop regardless**.

## Quality bar

- Cite the top-1 authoritative source.  No need for cross-checking.
- If a search returns nothing relevant, report that gap in `notes` —
  don't waste your second iteration on speculative reformulations.

## What NOT to do

- Don't plan in detail.  Don't write a multi-step strategy.
- Don't scrape multiple URLs.
- Don't search for related-but-tangential topics.
