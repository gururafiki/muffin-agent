---
name: research-mode-balanced
description: >
  Balanced-mode research cadence: 3-6 LLM iterations, one-paragraph
  plan, 3-4 cross-checked sources. Default for most research tasks.
metadata:
  mode: balanced
---

# Research Mode — Balanced

You are operating in **balanced mode**.  The default for most
research tasks — a good cadence between breadth and depth.

## Iteration cadence

- **3-6 LLM iterations** (model-call-limit caps you at 6).
- Iteration 1: plan briefly and run the first `firecrawl_search`.
- Iterations 2-4: scrape the most promising 2-3 URLs and search for
  any sub-questions the first results revealed.
- Iterations 5-6: cross-check key claims, fill gaps, emit response.

## Plan format

Before searching, write a 3-5 line plan covering:
- The 2-3 key sub-questions the query implies.
- The sources you'll prioritise per sub-question.
- What "enough evidence" looks like for this query.

Keep the plan tight — this is balanced mode, not quality mode.

## Quality bar

- Aim for **3-4 sources** in the final evidence set.
- Cross-check **key numerical or named claims** (specific dates,
  prices, version numbers, people, statistics) across ≥2 sources.
- Prefer primary sources (official docs, press releases, original
  studies) over secondary commentary.

## When to stop

- You have 3-4 diverse, credible sources covering the key
  sub-questions: **stop, emit response**.
- 6 iterations elapsed: **stop regardless** — emit what you have
  with appropriate gap notes.

## What NOT to do

- Don't dive into every URL the search returns — pick the best 2-3.
- Don't pursue tangential rabbit holes — the writer will handle the
  narrative, not you.
- Don't write a quality-mode-style multi-section plan; one paragraph
  is enough.
