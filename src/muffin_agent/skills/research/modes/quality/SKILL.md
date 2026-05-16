---
name: research-mode-quality
description: >
  Quality-mode research cadence: up to 25 LLM iterations, full 7-step
  Vane strategy, explicit contradiction-hunting, multi-source
  verification. Use for high-stakes or contested topics.
metadata:
  mode: quality
---

# Research Mode — Quality

You are operating in **quality mode**.  Optimise for **depth,
correctness, and source diversity** over speed.

## Iteration cadence

- **Up to 25 LLM iterations** (model-call-limit caps you at 25).
- Use the full budget if needed — the writer will produce a
  2000+ word output and needs rich evidence.
- Plan first.  Search broadly.  Scrape deeply.  Cross-check
  exhaustively.

## The 7-step research strategy

Cover each step explicitly for the topic at hand:

1. **Definition** — what is the subject; what are its boundaries?
2. **Features / characteristics** — primary attributes, mechanisms,
   structure.
3. **Comparisons** — how does it relate to / differ from alternatives?
4. **Recent developments** — news, updates, version changes,
   announcements from the last 12 months.
5. **Opinions** — what do experts / users / critics say?  Include
   both supportive and critical perspectives.
6. **Use cases** — concrete applications, customer stories, real-world
   adoption.
7. **Limitations** — known issues, caveats, edge cases, controversies.

If a step doesn't apply to the query (e.g. "use cases" for a
historical event), say so explicitly in `notes`.

## Plan format

Before searching, write a structured plan covering:
- The 5-10 sub-questions the query implies.
- A priority order (which to research first).
- A mapping from sub-questions to source types and search queries.
- Explicit stopping criteria for THIS query.

## Quality bar

- Aim for **6+ sources** in the final evidence set, **diverse** in:
  - **Type**: primary docs, news, opinion pieces, academic papers,
    discussion forums (whichever are allowed).
  - **Stance**: include both supportive and critical sources for
    contested topics.
  - **Recency**: include recent (<6 months) AND foundational
    (>1 year) sources where relevant.
- **Cross-check every load-bearing claim** across ≥2 independent
  sources.  If sources contradict, capture BOTH in evidence chunks
  and flag in `notes`.
- **Hunt for contradictions explicitly** — search for "criticism of
  X", "problems with X", "Y vs X" to surface dissenting views.

## When to stop

- You've covered all 7 strategy steps with multi-source evidence: stop.
- 25 iterations elapsed: stop, emit what you have, flag remaining
  gaps in `notes`.

## What NOT to do

- Don't stop early just because the first 2 sources agree — verify
  with a third.
- Don't accept one-sided coverage — explicitly seek dissent.
- Don't pad with low-quality sources just to hit a count — quality
  matters more than quantity even here.
