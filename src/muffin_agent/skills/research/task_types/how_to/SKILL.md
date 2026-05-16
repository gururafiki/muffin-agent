---
name: research-task-how-to
description: >
  How-to task type: gather concrete, ordered steps from authoritative
  documentation, plus prerequisites and common pitfalls.
metadata:
  task_type: how_to
---

# Task Type — How-To

The writer will produce a **numbered step-by-step guide** with
prerequisites and common pitfalls.  Your evidence should give the
writer concrete commands, config snippets, and the exact ordering.

## What to gather

1. **Prerequisites** — what does the user need installed / configured
   first?  (Versions, accounts, permissions, hardware.)
2. **The ordered steps themselves** — exact commands or UI actions,
   in the order they must be performed.
3. **Common pitfalls** — known failure modes, error messages people
   hit, "if you see X, the cause is usually Y".
4. **Verification** — how the user knows the procedure worked.

## Source priorities

1. **Official docs** — almost always the right source.  Pin to the
   version the user is likely on (or, if unknown, the latest stable).
2. **Maintainers' blog posts / release notes** — for newer changes
   not yet in docs.
3. **Stack Overflow / GitHub issues** — for the common-pitfalls
   section.  These are explicitly *welcome* for this task type even
   though they're community sources.
4. **Tutorials from credible authors** — only when official docs are
   incomplete.

## Evidence chunks

- Scrape the canonical doc page in full.  Don't rely on snippets —
  the writer needs the exact commands.
- For each pitfall, include the error message verbatim in the
  ``content`` field if you find it.
- Set ``source_type`` accurately: ``official_docs``, ``community``,
  ``blog``.

## Common pitfalls

- Including outdated steps from old tutorials — always cross-check
  against current official docs.
- Skipping prerequisites the writer would have no way to surface —
  if the official docs assume a Python venv, capture that.
- Missing the verification step — users want to know "did it work?"
