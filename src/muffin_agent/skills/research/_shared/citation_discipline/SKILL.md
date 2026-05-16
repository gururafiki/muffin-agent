---
name: research-citation-discipline
description: >
  Universal citation discipline for the research agent. Always
  applies; gathers high-quality, attributable evidence with explicit
  source provenance and surfaces uncertainty rather than hiding it.
---

# Citation Discipline (universal)

These rules **always apply**, regardless of mode or task_type.

## The cardinal rule

**Every claim in the final answer must be traceable to a specific URL
in your evidence list.**  If the writer can't cite it, you didn't
gather it well enough.

## What this means for evidence gathering

1. **No hallucinated URLs.**  If a tool didn't return a URL, do not
   invent one.  If you remember a URL from training data, do not
   include it without verifying via a tool call.

2. **Source provenance per chunk.**  Set ``title``, ``url``,
   ``source_type``, and ``retrieved_at`` on every ``EvidenceChunk``.
   These flow through to the writer's citation list.

3. **Prefer primary sources.**  If you cite a statistic, find the
   study / report / filing it came from, not a news article that
   re-reports it.  Secondary sources are fine *in addition*, not
   *instead of*.

4. **Date-check.**  For time-sensitive claims (current state,
   pricing, leadership, version numbers), confirm the source is
   recent.  Include the publication date in the ``snippet`` if it's
   not obvious from the URL.

5. **Attribute opinions to identifiable authors.**  "X is good" with
   no author is useless.  "X is good, according to <named expert /
   publication>" is citable.

## What this means for surfacing uncertainty

- If sources **contradict**, capture both perspectives in separate
  chunks.  Flag the contradiction in your ``notes``.
- If you **can't find** evidence for a sub-question, list it in
  ``notes`` — the writer will surface this in
  ``missing_information``.
- If the **only** evidence is community / opinion / low-credibility,
  flag this in ``notes`` so the writer can lower the final
  ``confidence``.

## What the writer will do with your evidence

The writer assigns ``[N]`` citation numbers to your chunks (in the
order it cites them), embeds them inline in the answer markdown, and
populates the ``sources`` list with matching ``n``/``title``/``url``
entries.  You don't assign numbers — just supply solid chunks.
