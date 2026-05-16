---
name: research-task-research-report
description: >
  Research-report task type: gather evidence suitable for a
  multi-section, comprehensive write-up with clear sub-topics and a
  limitations section.
metadata:
  task_type: research_report
---

# Task Type — Research Report

The writer will produce a **sectioned, comprehensive answer** with
clear ``## Heading`` markers and a ``## Limitations`` section.  Your
evidence should support that structure.

## What to gather

- **Definitions and scope**: what is the subject, what are its
  boundaries?
- **Sub-topic coverage**: identify 3-5 sub-topics the query implies
  and gather evidence for each.  Examples:
  - "Tell me about pgvector" → sub-topics might be installation,
    indexing, performance, ecosystem.
  - "What is the state of LLM agents?" → frameworks, use cases,
    benchmarks, open problems.
- **Recent developments**: news / version updates / announcements
  from the last 12 months.
- **Limitations and trade-offs**: known issues, caveats, criticisms.
  The writer's ``## Limitations`` section depends on this.

## Source priorities

1. **Primary sources** — official documentation, GitHub READMEs,
   press releases, academic papers, regulatory filings.
2. **High-quality secondary** — well-regarded blogs, news outlets
   with named authors, technical write-ups.
3. **Community signal** — only when relevant (e.g. adoption
   conversations); never as the sole source for factual claims.

## Evidence chunks

- One chunk per source.  Include enough ``content`` for the writer
  to extract sub-topic detail (not just a snippet).
- Use ``source_type`` accurately: ``web`` is the default; use
  ``academic``, ``news``, ``regulatory``, etc. when applicable.
- Date-stamp every chunk via ``retrieved_at``.

## Common pitfalls

- Over-indexing on one sub-topic — a research report needs balanced
  coverage.
- Ignoring the limitations dimension — the writer can't fabricate
  ``## Limitations`` content out of thin air.
- Cherry-picking only positive coverage — include critical
  perspectives where they exist.
