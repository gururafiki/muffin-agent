# Plan — Improve Agent Reliability & Efficiency

## Context

LangSmith run `019dc4bc-ced5-7ad2-ae2f-413d89d46ac3` (graph: `stock_evaluation`, ticker: AMZN) ran for **42.8 minutes** and ended in failure. Forensics show the slowness is not a single bug — it is a **cluster of compounding reliability issues** that turn one bad model choice into a 40-minute timeout cascade. This plan documents the root causes and proposes a layered set of fixes (model, middleware, tool routing, subagent contract, prompt) so the system degrades gracefully instead of collapsing.

---

## Root causes (evidence from trace)

| # | Root cause | Evidence | Time impact |
|---|------------|----------|-------------|
| **1** | **Free LLM is slow + corrupt** (`nvidia/nemotron-3-super-120b-a12b:free`) | 81 LLM calls × **31s avg** = 42 min of LLM latency. Output corrupted: `finish_reason="tool_callstool_calls"`, `model_name="…:freenvidia/…:free"` (chunks concatenated, never recovered). | ~80% |
| **2** | **`ModelRetryMiddleware` never wraps subagent LLM calls** | All 5 LLM errors are mid-stream `openai.APIError("Provider returned error")` exactly 120s long. Trace middleware chain at depths 2-7: `TodoListMiddleware → FilesystemMiddleware → SubAgentMiddleware → _DeepAgentsSummarizationMiddleware → AnthropicPromptCachingMiddleware → MemoryMiddleware`. **No `ModelRetryMiddleware`** in the trace, despite [agent_builder.py:503-515](src/muffin_agent/utils/agent_builder.py#L503-L515) registering it as outermost. CLAUDE.md documents this is supposed to handle exactly this error. | ~10% (5 × 120s = 600s wasted on un-retried failures) |
| **3** | **MCP tool calls fail with predictable, fixable errors that the LLM cannot reason about** | `equity_estimates_forward_sales`: 12/12 (100%) fail. `equity_estimates_forward_eps`: 8/12 fail. `equity_fundamental_*`: ~50% fail. Three failure families: (a) `Missing credential 'intrinio_api_key'` — provider not configured, (b) `Unauthorized FMP request -> 402 limit must be 0-5` — provider tier limit, (c) `Input should be less than or equal to 5, input: 10` — LLM hallucinated invalid args, (d) `No estimates data was returned for: AMZN` — provider has no data. Each failed tool call = 1 wasted 31s LLM round-trip. | ~5–10% |
| **4** | **Subagent retry storm at the orchestrator level** | Parent stock_evaluation re-asked `equity-estimates` 3× over 14 minutes after each subagent reported partial/failed data. `equity-fundamentals` was called 5×. The parent has no usable failure contract — the subagent returns prose, the parent re-issues a smaller variant. | ~10–15% |
| **5** | **Tool schemas don't communicate provider tier limits** | Tool docstrings (auto-generated from MCP) list `limit` as integer but don't say *current FMP free tier max is 5*. The LLM keeps trying `limit=10`. This is the same data the 422 error message contains — but only after the failure. | (multiplier on #3) |
| **6** | **`is_permanent_error` heuristic is too coarse** | [tool_error_handler/middleware.py:15-25](src/muffin_agent/middlewares/tool_error_handler/middleware.py#L15-L25) treats only exact-arg duplicates as cached. A 422 with `limit=10` does not stop the LLM from trying `limit=8` and getting 422 again. And `502 Bad Gateway` from FMP that says "limit must be 0-5" is treated as transient, not informative. | (multiplier on #3) |
| **7** | **Large MCP tool outputs flood context** | Avg 15k prompt tokens/call, max 72k. The 5-year cashflow JSON in the trace is ~30 KB embedded in messages. The summarization middleware exists but isn't kicking in soon enough — by call 30+ the parent has accumulated all subagent outputs. | small but compounds latency |

Tax these together: even with a fast model, items 3/4/5/6 alone would burn 5–10 minutes per stock evaluation.

---

## Improvement options

Grouped by layer; each item is independent and can ship separately.

### A. Model layer — biggest single lever

**A1. Stop using `:free` models as the default for production graphs.** *(REQUIRED)*
- Change [`model_config.py:9`](src/muffin_agent/model_config.py#L9) `DEFAULT_MODEL = "openai/gpt-oss-120b:free"` to a paid, stable model. Recommended: `anthropic/claude-sonnet-4-6` or `openai/gpt-4.1-mini`. Use `claude-haiku-4-5` for cheap data-collection subagents.
- Free OpenRouter routes have aggressive rate limits, low concurrency caps, frequent mid-stream cuts, and the Nemotron route has a known chunk-merging bug (visible in trace as `:freenvidia/...:free`).

**A2. Tiered model assignment per agent role.** *(RECOMMENDED)*
- Orchestrators (stock_evaluation, criterion_evaluation, criteria_definition) → strong model (Claude Sonnet 4.6 / GPT-4.1).
- Data-collection ReAct subagents → fast model (Haiku 4.5 / GPT-4.1-mini). They mostly route inputs to MCP tools.
- Reasoning-only nodes (data_validation, valuation, risk_assessment) → strong model.
- Implement via a per-agent `ModelConfiguration` override in `MuffinAgentBuilder`, e.g. `model_role: Literal["orchestrator", "collector", "reasoner"]` that picks from configured roles. This isolates the choice from the call site.

**A3. Tighten timeouts.** *(RECOMMENDED)*
- The 120 s wait per failed call points at `httpx`'s default. Add `timeout=60` to `ChatOpenAI`/`ChatAnthropic` in `get_llm()`; combined with `ModelRetryMiddleware` retries, total wall time per LLM step caps at ~3-4 minutes worst case instead of >5.

### B. Retry & error handling middleware

**B1. Verify `ModelRetryMiddleware` actually fires; fix if not.** *(REQUIRED)*
- Reproduce locally with a synthetic mid-stream `openai.APIError` and confirm retries happen.
- Suspect: `create_deep_agent` may be reordering or not honoring the user-supplied middleware position. The deep agent installs its own `_DeepAgentsSummarizationMiddleware`, `SubAgentMiddleware`, `TodoListMiddleware`, `MemoryMiddleware` *outside* user middleware, putting `ModelRetryMiddleware` *inside* and (depending on the framework's wrap direction) potentially making it a no-op for top-level errors. Check `langchain.agents.middleware.ModelRetryMiddleware` source — it should hook `awrap_model_call` (per-LLM, innermost-friendly) but may need `on_failure="retry"` not `"error"`, or may need a different placement.
- Alternative if upstream middleware can't be made to fire reliably: subclass `ChatOpenAI` / wrap at the model level so retry is bound to the LLM client itself, not the agent middleware stack. Tenacity around `_astream` is the simplest implementation.

**B2. Increase `max_retries` and broaden retry triggers.** *(QUICK WIN)*
- Current: 3 retries × 30s max delay = sufficient *if* it fires. Bump to 5, and add `httpx.RemoteProtocolError`, `httpx.ReadTimeout`, and OpenRouter-specific JSON-error-with-200 to the retry-on list. The 5 errors in the trace would be retry-eligible.

**B3. Smart `ToolErrorHandlerMiddleware` — error-aware, not just args-aware.** *(RECOMMENDED)*
- Today: blocks only *identical* (tool, args) repeats and only for substring-matched "permanent" errors.
- Upgrade to **error-class learning**: when `equity_fundamental_cash` returns `422 limit must be ≤ 5`, store the constraint *for the tool*, not the (tool, args) pair. Inject the learned constraint as a `ToolMessage` hint on the next call: `Note: previous call to equity_fundamental_cash failed because limit must be ≤ 5`.
- Three error families to detect:
  1. `HTTP 422 ... Input should be less than or equal to N` → extract bound, surface to LLM, optionally clip args before call.
  2. `Unauthorized {provider} request -> 402 ... limit must be between 0 and N` → same.
  3. `Missing credential '{provider}_api_key'` → mark *that provider* as unavailable for the session, so the LLM doesn't pick it again. Today the LLM just retries with the same default provider.

**B4. Provider routing fallback.** *(RECOMMENDED)*
- For OpenBB MCP tools where `provider` matters (most `equity_*` tools), build a small `ProviderRouterMiddleware` that:
  - Maintains a per-session "known-bad provider for tool X" set populated from B3.
  - If LLM calls with a known-bad provider → rewrite to the next configured provider (e.g. `intrinio` → `fmp` → `yfinance`) before dispatch, with a one-line `ToolMessage` annotation saying so.
  - Eliminates the most common error (LLM picks `intrinio`, no key, fails).

### C. Tool & MCP layer

**C1. Wrap MCP tools with constraint-aware schemas.** *(RECOMMENDED)*
- After loading MCP tools in [`data_collection/utils.py`](src/muffin_agent/agents/data_collection/utils.py), apply a small adapter that:
  - Reads tool name, looks up known constraints (max `limit` per provider tier, supported `period` values, supported `provider` list given configured API keys).
  - Rewrites the tool's `args_schema` to include those bounds (`Field(le=5, description="Max 5 on FMP free tier")`) and the docstring.
  - The LLM sees the constraint *before* its first call instead of learning it from a 422.

**C2. Default-provider per agent.** *(RECOMMENDED)*
- Each data-collection agent declares a *preferred provider order* matching the keys actually configured in `.env` (e.g. equity_fundamentals: `["fmp", "yfinance"]`, never `intrinio`). The agent prompt lists these explicitly. The router from B4 enforces them.
- Removes a whole class of errors at zero LLM cost.

**C3. Pre-flight credential check at startup.** *(NICE TO HAVE)*
- On agent build, ping each provider once and log which are reachable. Surface this as a warning in the system prompt: `Available providers: fmp, yfinance. Do NOT use: intrinio, polygon (no API key)`.

**C4. Aggressive output truncation for MCP tool results.** *(RECOMMENDED)*
- `ToolResultCacheMiddleware` already offloads results to the store. Add an automatic **summarize-on-write** when a cache entry > N tokens: store full JSON in `/data/cache/`, return only the schema-summary as the `ToolMessage` body. The LLM uses `read_sandbox_file_to_store` only when it actually needs raw rows. Cuts the 30 KB cashflow JSON (and similar) out of the conversation.
- LangGraph's built-in summarization middleware works on assistant messages, not tool messages — this is a complementary gap.

### D. Subagent contract & orchestration

**D1. Subagent must return structured JSON, not prose.** *(REQUIRED)*
- Today subagents return free text → parent re-asks because it can't tell partial-success from failure. Each subagent gets `response_format=AutoStrategy(schema=DataCollectionResult)` where the schema includes `requested_fields`, `retrieved_fields`, `missing_fields`, `errors_per_tool`, `data_payload`, `notes`.
- Parent reads `missing_fields` and decides whether to re-call (and what to ask for) instead of guessing from prose.

**D2. Subagent budget caps.** *(REQUIRED)*
- Each `task` invocation gets `max_iterations` and `max_tool_calls` ceilings (e.g. 10 LLM steps, 15 tool calls). When hit, return a structured "budget exhausted, here's what I have" instead of looping.
- Today nothing stops a subagent from spinning 30+ ReAct steps on a flaky provider.

**D3. Idempotent subagent re-call detection.** *(RECOMMENDED)*
- Hash (subagent_name, normalized_description) per call. If parent re-issues a similar request twice, return the prior result with a hint: `"You already received data for this query. If the prior data was insufficient, refine the description to specify what's missing."`
- Caps the orchestrator retry storm visible in the trace (3× same `equity-estimates` ask).

**D4. Pre-fanout planning step.** *(NICE TO HAVE)*
- The deep agent's ReAct loop intersperses planning with tool calls. For data-heavy graphs (stock_evaluation), a dedicated **planning node** drafts the full subagent call list up front (using the strong model once), then a fanout node dispatches them in parallel. Reduces serial latency dramatically — currently the 10 task calls were strictly sequential. Available subagent invocations in this run actually had no inter-dependencies.

### E. Prompts & system behavior

**E1. Provider/limit guidance in data-collection prompts.** *(QUICK WIN)*
- Append a per-agent block: `"Available providers: fmp (limit≤5 on free tier), yfinance (no limit param). Default to yfinance for >5y history. Never call with provider=intrinio."`
- This is what C1/C2 enable; the prompt is the cheapest place to land it.

**E2. Tighten the orchestrator stock_evaluation prompt.** *(RECOMMENDED)*
- Forbid re-asking the same subagent without naming the specific missing field. Pair with D1.
- Forbid re-asking after 2 attempts; mark the dimension `data_unavailable` instead.

**E3. Reduce ReAct chatter.** *(QUICK WIN)*
- Many of the 81 LLM calls were "thought-only" ReAct steps with no tool call. With a small/cheap model these are fine; with the strong orchestrator model they cost dollars and seconds. Add `max_consecutive_assistant_messages=1` (or equivalent) so the agent must call a tool or finish.

### F. Observability

**F1. Add a session-level dashboard panel: tool-error-rate, LLM-retries, subagent-iterations.** *(NICE TO HAVE)*
- Surface these via LangSmith metadata so trace queries like `tool_error_rate > 0.5` find runs like this one before they're discovered manually.

---

## Recommended sequencing (smallest blast radius first)

| Wave | Items | Why first |
|------|-------|-----------|
| **1 — Today** | A1 (drop free model), A3 (timeouts), B2 (retries config), E1 + E3 (prompts) | Pure config / prompt changes. Likely cuts runtime 5-10× alone. No structural risk. |
| **2 — This week** | B1 (verify retry middleware), B3 (smart error handler), C1 (constraint-aware schemas), D1 (structured subagent output) | Code changes but localized. Each fixes a documented failure family from the trace. |
| **3 — Next** | A2 (tiered models), B4 (provider router), C2 (per-agent provider defaults), C4 (auto-summarize tool results), D2 (budget caps) | Structural. Each requires a small new abstraction or middleware. |
| **4 — Later** | D3 (idempotent re-call), D4 (pre-fanout planning), C3 (preflight check), F1 (dashboard) | Nice to have once the above is stable. |

---

## Critical files / functions referenced

- [src/muffin_agent/model_config.py:9](src/muffin_agent/model_config.py#L9) — `DEFAULT_MODEL`; change for A1.
- [src/muffin_agent/model_config.py:75-123](src/muffin_agent/model_config.py#L75-L123) — `get_llm()`; add timeout for A3.
- [src/muffin_agent/utils/agent_builder.py:475-529](src/muffin_agent/utils/agent_builder.py#L475-L529) — `_assemble_middleware`; verify ModelRetryMiddleware position for B1.
- [src/muffin_agent/utils/agent_builder.py:99-103](src/muffin_agent/utils/agent_builder.py#L99-L103) — `_should_retry_llm_call`; broaden for B2.
- [src/muffin_agent/middlewares/tool_error_handler/middleware.py](src/muffin_agent/middlewares/tool_error_handler/middleware.py) — extend for B3 (error-class learning).
- [src/muffin_agent/middlewares/tool_result_cache/middleware.py:140](src/muffin_agent/middlewares/tool_result_cache/middleware.py#L140) — auto-summarize on cache write for C4.
- [src/muffin_agent/agents/data_collection/utils.py](src/muffin_agent/agents/data_collection/utils.py) — adapter site for C1, C2, B4.
- [src/muffin_agent/agents/data_collection/equity_estimates.py](src/muffin_agent/agents/data_collection/equity_estimates.py) and the other 13 collection agents — receive D1 structured output.
- [src/muffin_agent/agents/subagents.py](src/muffin_agent/agents/subagents.py) — wire D2 budget caps into `CompiledSubAgent` construction.
- [src/muffin_agent/prompts/data_collection/equity_estimates.jinja](src/muffin_agent/prompts/data_collection/equity_estimates.jinja) and siblings — E1.
- [src/muffin_agent/prompts/stock_evaluation.jinja](src/muffin_agent/prompts/stock_evaluation.jinja) — E2.

## Verification

- **Smoke**: re-run AMZN stock_evaluation on a paid model (after Wave 1). Expect <8 minutes, no `APIError` failures, all data dimensions populated or marked `data_unavailable`.
- **Unit**: tests for B3 error-class extraction (parametrize on the 3 error families seen in trace), C1 schema rewriting, D1 schema validation.
- **Integration**: spin up a thread, force `equity_fundamental_balance` to fail with 422, assert `ToolErrorHandlerMiddleware` injects the constraint hint and the next call respects it.
- **LangSmith**: filter recent traces for `--min-latency 600 --error` after each wave; expect count to drop monotonically.
- **Memory**: log to CLAUDE.md the "free model is forbidden as default" lesson and the "MCP tool docstrings need provider-tier constraints" lesson.

## Open questions for the user

1. Which paid model is the budget for? (Claude Sonnet 4.6 vs GPT-4.1 vs OpenRouter paid Nemotron — affects A1.)
2. Are Intrinio / Polygon / FMP-paid keys planned, or should `intrinio` be permanently de-listed in agent prompts? (Affects C2.)
3. Is the `task` subagent budget cap (D2) acceptable as a soft default, or should it be configurable per subagent type?
4. Which wave wants implementation now — Wave 1 only, or 1+2 bundled?
