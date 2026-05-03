# Diagnosis: web_search loop on run `019dc3fd-4b1e-73f0-863c-febc4555dda7`

## Context

You reported that the `web_search` data-collection ReAct agent looped on tool calls and that
tool results came back as `Tool result too large, the result of this tool call ...`. I pulled
the trace from LangSmith (project `Muffin-test`, 247 child runs, 30 LLM calls, ~8 min wall
time) and reproduced the failure mode in the message history. This document is an
investigation summary, not an implementation plan — the goal is to align on root cause and
a fix direction.

## What actually happened

**Task**: `Search for the latest tweet from Amazon's main Twitter/X account (@amazon).
Provide tweet text, timestamp, and URL.`

**Model**: `nvidia/nemotron-3-super-120b-a12b-20230311:free` (OpenRouter free tier).
The trace shows doubled metadata in every AI response — `finish_reason="tool_callstool_calls"`,
`model_name="...:freenvidia/...:free"`. This is a bug in the free-tier endpoint and a strong
signal the model is unreliable.

**Loop pattern** (e.g. LLM run `019dc3ff-1ed7-7933-8423-e35714871922`, messages `[12]→[15]`):

1. msg `[12]` AI → `firecrawl_search(query="from:amazon", scrapeOptions.markdown=true, limit=5)`
2. msg `[13]` ToolMessage (1010 chars):
   ```
   Tool result too large, the result of this tool call chatcmpl-tool-9b26331c8ae0c87a
   was saved in the filesystem at this path: /large_tool_results/chatcmpl-tool-9b26331c8ae0c87a
   ...you can read the result from the filesystem by using the read_file tool...
   ```
3. msg `[14]` AI → **does NOT call `read_file`**. Fires another `firecrawl_search` with a
   different query (`AMZN site:x.com`).
4. Loop continues for ~30 LLM calls until the run was killed.

**Every single AIMessage in the trace has `content` of length 0** — only tool calls,
zero assistant prose. The model never produces reasoning, never decides to give up, never
acknowledges what it just saw.

## Root cause (layered)

1. **Free-tier model is broken at tool-using.** Doubled response metadata + always-empty
   text content + ignoring the offload instruction = the model is the proximate cause.

2. **FilesystemMiddleware offload triggers easily here.** Default
   `tool_token_limit_before_evict = 20000` tokens × 4 chars/token = 80,000 chars
   ([deepagents/middleware/filesystem.py:581](.venv/lib/python3.13/site-packages/deepagents/middleware/filesystem.py#L581),
   [filesystem.py:1340-1395](.venv/lib/python3.13/site-packages/deepagents/middleware/filesystem.py#L1340-L1395)).
   `firecrawl_search` with `scrapeOptions.formats=["markdown"]` returns full page Markdown
   for every result and blew past 80K chars in this run.

3. **The recovery instruction is far from the model's eye-line.** The "use `read_file` on
   `/large_tool_results/<tool_call_id>`" instruction lives only in
   `FilesystemMiddleware`'s auto-injected snippet
   ([filesystem.py:319](.venv/lib/python3.13/site-packages/deepagents/middleware/filesystem.py#L319)).
   The web_search system prompt
   [src/muffin_agent/prompts/data_collection/web_search.jinja](src/muffin_agent/prompts/data_collection/web_search.jinja)
   does **not** mention offload, `read_file`, or `/large_tool_results/`. A weaker model has
   to stitch the two together at runtime.

4. **Firecrawl can't retrieve tweet content from x.com anyway.** X.com gates timelines
   behind auth; Firecrawl returns metadata pages (`{"markdown": "Don't miss what's
   happening. People on X are the first to know..."}`) but never actual tweet bodies.
   The task is unsolvable with the current tool surface — the model would loop even on a
   capable backbone, just less floridly.

5. **`firecrawl_extract` is misconfigured in the Firecrawl deployment.** Msg `[13]` of the
   final LLM call returned: `"OpenAI API key is missing. Pass it using the 'apiKey'
   parameter or the OPENAI_API_KEY environment variable."` So the model's one structured
   alternative also fails.

## Why this matters beyond this run

The same failure shape can hit any data-collection sub-agent if (a) a tool result crosses
80K chars and (b) the LLM doesn't reach for `read_file`. The web_search prompt is silent
on the recovery flow, so we're entirely reliant on the FilesystemMiddleware partial being
loud enough. With a strong model that's fine; with weak/free-tier models it isn't.

## Critical files referenced

- [src/muffin_agent/agents/data_collection/web_search.py](src/muffin_agent/agents/data_collection/web_search.py) — agent factory
- [src/muffin_agent/prompts/data_collection/web_search.jinja](src/muffin_agent/prompts/data_collection/web_search.jinja) — system prompt (no offload guidance)
- [src/muffin_agent/utils/agent_builder.py](src/muffin_agent/utils/agent_builder.py) — wires `FilesystemMiddleware` via `.with_short_term_memory()`
- `.venv/.../deepagents/middleware/filesystem.py:384` — `TOO_LARGE_TOOL_MSG` template
- `.venv/.../deepagents/middleware/filesystem.py:581` — `tool_token_limit_before_evict=20000` default

## Scope

**Diagnosis only — no code changes proposed in this plan.** The investigation above is
the deliverable. Future fix work should be guided by the principle below.

## Guiding principle for any future fix (per user)

Do NOT hardcode per-site/per-task guardrails (e.g. "don't try X.com"). The failure mode to
solve generically is: **when a task is unsolvable with the available tools, the agent must
notice and stop** — it must not repeat similar tool calls forever. Reliability of tool
calls and detection of unproductive loops are framework-level concerns, not per-prompt
patches.

That rules out the narrow prompt fixes (e.g. "X.com is auth-walled") and points toward
loop/repetition detection that works for any tool, any sub-agent.

## Plausible remediation directions (for a future planning session)

These are not part of this plan — listed only so the diagnosis is actionable later.

- **Loop / repetition detection middleware**: track recent tool calls per agent run; if
  the same tool is called >N times with semantically similar args and no progress (e.g.
  no new information observed, AIMessage content stays empty), force a terminal
  "data unavailable" response. This is the only direction that addresses the user's
  principle directly.
- **Generalize the offload-recovery instruction in the agent prompt** (not site-specific):
  "If a ToolMessage begins with `Tool result too large`, your next action MUST be
  `read_file` on the path it gives you. Do not retry the original tool." Lives in the
  agent prompt instead of relying solely on the FilesystemMiddleware auto-snippet.
- **Stop using the free-tier OpenRouter Nemotron** for sub-agents — its doubled metadata
  and empty assistant text are the proximate trigger. Pick any reliable tool-calling
  model as the default and keep the free-tier model opt-in for cost-sensitive runs.
- **Fix Firecrawl's `OPENAI_API_KEY`** so `firecrawl_extract` works (infra-level, not
  agent-level).
