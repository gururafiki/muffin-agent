# Plan — Adaptive, Provider-Agnostic Reliability for Muffin Agents

## Context

Trace `019dc4bc-ced5-7ad2-ae2f-413d89d46ac3` (stock_evaluation, AMZN) ran 42.8 min and failed. Forensics show the runtime is not one bug but the predictable behaviour of a system whose reliability is **bound to the worst-quality call in the loop**: when the model is slow, when one MCP provider is missing a key, or when a tool returns a 422, the agent has no way to *learn from it within the run* — so it pays the same penalty again on the next subagent, the next iteration, the next ticker.

The user's requirement is therefore **architectural, not tactical**:

> "I prefer LLM to learn based on past experience and improve future executions. I want a system that is robust and adaptive no matter how reliable provider/tool is."

Two further constraints from the user:
1. Avoid bespoke per-integration code (no hand-rolled `Field(le=5)` per OpenBB tool).
2. Prefer composable primitives already in `langchain.agents.middleware` / `deepagents` over custom plumbing where they fit.
3. Re-evaluate whether subagents are the right shape, including dropping them in favour of `LLMToolSelectorMiddleware`.
4. Re-evaluate context management generically (`ContextEditingMiddleware`, `SummarizationMiddleware`).
5. Make parent ↔ subagent communication actually bidirectional rather than re-spawning from scratch.

This plan follows that brief: every problem gets multiple options, an explicit recommendation, and the reason for the recommendation.

---

## Root causes (re-stated, with the adaptive lens)

| # | Symptom | Underlying defect | Why it matters here |
|---|---------|-------------------|---------------------|
| **R1** | 81 LLM calls × 31 s avg, 5 mid-stream `APIError` | Model layer has no retry/fallback that fires inside `awrap_model_call`. Trace shows no `ModelRetryMiddleware` span. | Without retry, every flaky-stream chunk-merge bug is a 120 s hang. |
| **R2** | Same wrong tool calls repeatedly (`limit=10` against `≤5`, `provider=intrinio` with no key, `forward_sales` 12/12 fail) | No **episodic tool-knowledge** carried within or across calls. Existing `ToolErrorHandlerMiddleware` only blocks *byte-identical* arg dictionaries; the LLM tries `limit=10`, then `limit=8`, then `limit=6`, paying a full LLM round-trip each time. | This is the single biggest *adaptive* gap. The LLM never reads back the lesson it just learned. |
| **R3** | Parent re-spawns `equity-estimates` 3× over 14 min, `equity-fundamentals` 5× | Subagent contract is one-shot prose; parent has no way to ask "you reported gaps — fill *just these* fields" so it re-issues the whole task. | Trades quality for tokens: every refinement is a fresh full subagent boot with full prompt + new tool calls. |
| **R4** | 1.15 M cumulative prompt tokens; 30 KB cashflow JSON sits in messages | Tool output spills into context; `ToolResultCacheMiddleware` caches but does not *trim* what the LLM sees. | Each subsequent LLM call re-tokenises the same payload. With slow models, this multiplies. |
| **R5** | `nvidia/nemotron-3-super-120b-a12b:free` chosen as the default | Default in `model_config.py:9`. Free OpenRouter routes have known mid-stream chunk-merge bugs (visible in trace: `model_name="…:freenvidia/…:free"`). | Floor on quality; the rest of the system inherits the floor. |
| **R6** | 14 narrow subagents, each booting its own MCP-loaded ReAct loop | Subagent boundaries set up *during exploration*, not *during use*. Most subagents only call 1–3 tools per spawn. | The work-per-spawn does not pay for the spawn cost. Stateless re-spawning amplifies R3. |
| **R7** | 42.8 min wall time despite no genuine inter-step dependency | Fully serial orchestration. The 10 task calls in the trace are commutative — they could have run as one parallel batch. | Latency is the user-facing pain. Parallelism dwarfs every other tactical fix. |

---

## Cross-cutting design principles for the rewrite

1. **Prefer episodic memory over schemas.** The system should not need a constraint declared per provider; it should *learn* the constraint the first time a provider rejects a call.
2. **Prefer middleware composition over bespoke logic.** `ToolRetryMiddleware`, `ModelRetryMiddleware`, `ModelFallbackMiddleware`, `ContextEditingMiddleware`, `LLMToolSelectorMiddleware`, `ToolCallLimitMiddleware`, `ModelCallLimitMiddleware` are all upstream-supported.
3. **Use existing storage (BaseStore + state)**, not new tables. The shared `InMemoryStore` already underpins `ToolResultCacheMiddleware`. Extend it.
4. **Subagents are an optimisation, not the architecture.** They earn their place only when they meaningfully isolate context. Otherwise tools live at the orchestrator level with `LLMToolSelectorMiddleware` filtering.
5. **Parent ↔ subagent communication is conversational.** A subagent reports `gaps`, the parent issues a *delta* request, the subagent reads its own prior cached output and fills gaps only.

---

## Problem 1 — Tool-failure learning (R2)

The LLM should never make the same mistake twice. The current `ToolErrorHandlerMiddleware` blocks byte-identical retries; that's not learning, it's a circuit breaker.

### Options

| Option | What it does | Tradeoffs |
|---|---|---|
| **1A. Generalise `ToolErrorHandlerMiddleware` into a `ToolKnowledgeMiddleware`.** Every tool error is parsed into a *lesson* object: `{tool, error_class, constraint_extracted, raw_message, occurred_at}`. Lessons are written to the shared store under `("tool_lessons", tool_name)`. Before each `ToolMessage` is rendered to the LLM (and before the model call), all relevant lessons are injected as a small "## Known constraints" block in the system prompt or as a synthetic `ToolMessage`. | Generic, no per-tool code. Lessons accumulate across runs (when store is persistent). LLM sees them in the same channel as data. **Cost**: prompt grows by the lesson count; mitigated by capping per-tool to N latest. |
| 1B. Modify the existing middleware to inject the *previous error message verbatim* on the next call to the same tool, regardless of args. | Cheapest. Works as a soft "remember the last failure". **Cost**: only learns from the most recent error; doesn't extract structure. The free-Nemotron model demonstrably forgets lessons one turn later. |
| 1C. Auto-rewrite tool args before dispatch using extracted constraints (e.g. clip `limit=10` → `limit=5`). | Strongest correction. **Cost**: dangerous because args have semantics — clipping `limit` is fine, clipping `period` is not. Hard to do generically. Reject. |
| 1D. Use `ToolRetryMiddleware` with a custom `on_failure` formatter that paraphrases the error into a "lesson" string. | Native, near-zero code. **Cost**: doesn't *persist* knowledge between turns; the lesson is in a single ToolMessage that may scroll out of context. |
| 1E. Combine 1A + 1D: `ToolRetryMiddleware` for one-shot retry with paraphrase; `ToolKnowledgeMiddleware` for persistence. | **Recommended.** Retry handles transients (1D); knowledge handles durable lessons (1A). |

### Recommendation: **1E**

Implementation sketch (no per-tool code):

```python
class ToolKnowledgeMiddleware(AgentMiddleware):
    """Learn from tool failures and surface lessons to the LLM."""

    # Pluggable parsers. Each receives (tool_name, error_msg) and returns a
    # short, action-oriented lesson string or None.
    _PARSERS: list[Callable[[str, str], str | None]] = [
        _parse_pydantic_constraint,    # "limit must be ≤ 5"
        _parse_provider_quota,         # "FMP free tier: limit ≤ 5 for fundamentals"
        _parse_missing_credential,     # "Provider 'intrinio' is not configured — do not use"
        _parse_no_data,                # "Provider 'fmp' has no estimates for AMZN — try yfinance"
        _parse_generic_http,           # fallback: include the verbatim message
    ]

    async def awrap_tool_call(self, request, handler):
        result = await handler(request)
        if isinstance(result, ToolMessage) and is_error_content(result.content):
            await self._record_lesson(request, result.content)
        return result

    async def abefore_model(self, state, runtime):
        lessons = await self._load_lessons_for_tools(runtime)
        if lessons:
            return {"tool_lessons_block": render(lessons)}  # injected via dynamic_prompt

    @dynamic_prompt
    async def system_addition(self, state, runtime) -> str:
        return state.get("tool_lessons_block", "")
```

The parsers are 5 small regex/substring functions — *not* per-tool, but per-error-shape. They cover the four families seen in the trace and degrade gracefully (1E falls back to verbatim).

Lessons live in the shared `BaseStore` so:
- All subagents see the same lessons.
- Persists across LangGraph threads when a persistent store is wired.
- Survives `execute_python` sandbox recycling.

The block injected to the prompt looks like:

```
## Tool lessons learned this session
- equity_fundamental_balance: limit must be ≤ 5 (FMP free tier).
- equity_estimates_forward_eps: provider 'intrinio' is not configured. Do not pass provider=intrinio.
- equity_estimates_forward_sales: returns "No estimates data" for AMZN. Skip this tool for AMZN; try equity_estimates_consensus instead.
```

This is the *cheapest possible adaptation surface* and exactly what Reflexion / Voyager-style agents do.

---

## Problem 2 — Provider-, model-, and stream-level resilience (R1, R5)

A free, mid-stream-flaky model with no retry is the dominant runtime cost.

### Options

| Option | What it does | Tradeoffs |
|---|---|---|
| **2A. Layer `ModelRetryMiddleware` + `ModelFallbackMiddleware` natively.** Retry on transient/streaming errors; if all retries exhausted, fall back to a different model entirely (e.g. paid Sonnet 4.6 → free Nemotron, or Haiku → Sonnet). | Ships today, no code. **Cost**: requires a fallback model configured. Requires verifying retry actually fires (it didn't in the trace — needs investigation in `_assemble_middleware` placement). |
| 2B. Attach retry at the `ChatOpenAI` constructor level via `tenacity` around `_astream`. | Bypasses any middleware-ordering issue. **Cost**: hides retries from LangSmith trace; harder to debug. |
| 2C. Use the LangChain primitive `ChatModel.with_fallbacks([…])` instead of middleware. | Native. **Cost**: doesn't compose with middleware; loses prompt-level visibility. |
| **2D.** Combine 2A with **per-role default models** in `ModelConfiguration`: orchestrators get strong/paid; data-collection ReAct loops get fast/cheap; fallback chain Sonnet→Haiku→Nemotron-free covers cost vs. availability. | **Recommended.** 2A's robustness with cost discipline. Composes with everything else. |
| 2E. Adopt OpenRouter's own provider-routing (`provider: { order: […], allow_fallbacks: true }`) and pin paid tier in default. | Uses provider's native fallback. **Cost**: still a single model call from langchain's perspective; no LangChain-level visibility into the fallback. |

### Recommendation: **2D + verify 2A is firing**

- Step 1: write a failing unit test that injects a synthetic mid-stream `APIError` into a stub `ChatOpenAI._astream` and asserts the agent sees a retried response. Trace currently shows no retry span, so this test would fail on `main`. Fix wiring inside `MuffinAgentBuilder._assemble_middleware` (likely a placement issue with deep-agent built-in middleware ordering — `create_deep_agent` may hoist its own middleware ahead of user middleware; if so, register `ModelRetryMiddleware` via `with_middleware()` *and* `with_fallback_models()` so they enter the user-middleware slot which is honoured).
- Step 2: introduce `ModelConfiguration.role` (`"orchestrator" | "collector" | "reasoner"`) and per-role model lists.
- Step 3: introduce `ModelConfiguration.fallback_models` (ordered list); `MuffinAgentBuilder` wires `ModelFallbackMiddleware(*fallbacks)` automatically when populated.

Trace would have completed in roughly half the time even with the same Nemotron primary, just by retrying the 5 streaming errors and falling back to Haiku once Nemotron's chunk merging broke.

---

## Problem 3 — Context bloat (R4)

Tool outputs flood the conversation; the prompt grows unboundedly; the slow model re-tokenises old payloads on every step.

### Options

| Option | What it does | Tradeoffs |
|---|---|---|
| **3A. Add upstream `ContextEditingMiddleware` with `ClearToolUsesEdit(trigger=40_000, keep=4)`.** | Native, configured once. Older tool messages auto-replaced with `[cleared]` placeholder, recent ones kept verbatim. Works with any tool — no per-tool code. | **Cost**: cleared messages can hide context the LLM still needs. Mitigated by `keep=N`. |
| 3B. Add upstream `SummarizationMiddleware` (full-conversation summary on token threshold). | Most aggressive size reduction. **Cost**: summarisation itself is an LLM call; with a free model this can be slow and lossy. Better as a fallback after 3A is exhausted. |
| **3C. Extend `ToolResultCacheMiddleware` to *summary-on-write*: whenever a cached entry exceeds N tokens, store the full payload in `/data/cache/<key>.json` (sandbox), but return a *schema-summary* `ToolMessage` (e.g. "5-year cashflow stored at /data/cache/...; columns = [...]; row_count=5"). LLM uses `read_sandbox_file_to_store` only when it actually needs raw rows.** | Generic, ZERO per-tool code. Plays with the existing offload pattern. **Cost**: requires the LLM to know the file is there; covered by an existing prompt partial. |
| 3D. Per-tool output truncation rules. | Rejected — bespoke per-integration code. |
| 3E. LangGraph `trim_messages` after each tool result. | Same effect as 3A but less granular. |

### Recommendation: **3A + 3C in parallel; keep 3B as fallback**

- 3A handles old tool messages indiscriminately at the message layer.
- 3C never lets a >20 K-token payload enter the message stream in the first place.
- 3B turns on only when token usage still climbs past 80 K (rare with 3A+3C). Adopt LangChain's exact-token-counter so it doesn't undershoot on Anthropic models.

This is fully generic — no per-MCP-tool code.

---

## Problem 4 — Subagent vs. flat-tool architecture (R6, R7)

The current 14 ReAct subagents were built so each domain has its own narrow tool set and prompt. The trace shows this trades correctness (each subagent works in isolation) for latency (each spawn boots a fresh ReAct loop) and rigidity (parent must re-issue full tasks for refinements).

### Options

| Option | What it does | Tradeoffs |
|---|---|---|
| **4A. Drop most narrow subagents. Move all OpenBB tools to the orchestrator. Use `LLMToolSelectorMiddleware(model=<haiku>, max_tools=8)` to filter the registry per turn.** | Aligns with user request. One agent loop, no spawn cost, no inter-agent retry storms. The selector picks 8 most-relevant tools per turn from ~80 OpenBB+Firecrawl tools. Selector LLM is small/cheap (haiku-class). | **Cost**: orchestrator system prompt grows with all domain guidance. Mitigated by dynamic prompt partials (only the rules for the *currently selected* tools are injected). **Cost**: loses the natural context isolation of subagents — if a single MCP tool returns 30 KB JSON, it's now in the orchestrator's history. Mitigated by 3C (auto-summary). |
| 4B. Keep subagents but redefine them coarser: 3 subagents (`market-context`, `company-fundamentals`, `peer-and-sentiment`) instead of 14. | Halfway. Less spawn overhead, retains some isolation. **Cost**: still N>1 spawn overhead for refinements; still need the bidirectional protocol. |
| **4C.** Hybrid: orchestrator has direct access to all *light, atomic* tools (price quote, single FRED series). Subagents survive only for **heavy contexts** (`equity-fundamentals` returns 30 KB JSON; `web-search` returns long crawls). Selector middleware filters within each layer. | **Recommended.** Captures 4A's flatness for cheap reads while keeping subagent isolation for the cases that genuinely benefit. |
| 4D. Replace subagents with parallel branches in the orchestrator graph (`langgraph.types.Send`). | Best raw latency. **Cost**: requires baking the dispatch into the graph, less LLM-driven. Good for screening pipelines (already used in `equity_screening.py`); awkward for free-form `stock_evaluation` which doesn't know the work-set up front. |
| 4E. Status quo. | Rejected — trace evidence is conclusive that the current shape is the dominant latency source. |

### Recommendation: **4C, with the hybrid line drawn at "tool output size"**

Operational rule: a domain becomes a subagent only if (a) any of its tool outputs typically exceed 5 K tokens (e.g. fundamentals JSON, web crawls, full options chains) **or** (b) it requires its own `system_prompt_template` to reason correctly (e.g. data_validation, forecasting).

By that rule, surviving subagents for `stock_evaluation`:
- `equity-fundamentals` (heavy JSON)
- `web-search` (heavy text)
- `data-validation` (specialist reasoning)
- `regulatory-filings` (long documents)
- `news` (variable but often heavy)

Dissolved (their tools move to the orchestrator with `LLMToolSelectorMiddleware`):
- `equity-price`, `equity-estimates`, `equity-ownership`, `economy-macro`, `fixed-income`, `etf-index`, `discovery-screening`, `currency-commodities`, `fama-french`, `options`.

This deletes ~10 subagent factories and roughly halves the number of LLM round-trips per stock — a single tool call replaces what was a `task → subagent ReAct loop → return` cycle.

---

## Problem 5 — Bidirectional parent ↔ subagent communication (R3)

Today the parent re-asks the same subagent because:
1. The subagent reply is prose; the parent has to *interpret* whether data was complete.
2. Each `task` invocation is a fresh stateless `ainvoke` — the subagent has no memory of its prior call.
3. The parent cannot say "fill these specific gaps"; it can only re-issue a description.

### Options

| Option | What it does | Tradeoffs |
|---|---|---|
| 5A. Subagent returns structured `DataCollectionResult`. Parent checks `gaps` and re-issues a smaller request mentioning the gaps. | Easy. **Cost**: still stateless; subagent re-explores its own prompt and tools each call. Just makes the gap visible. |
| **5B. "Conversational subagent": subagent caches its own structured output to `/scratch/subagent_runs/{call_id}.json` keyed on `task_call_id`. Parent's follow-up `task` call passes `prior_call_id` as part of the description; the subagent prompt teaches it to read its prior output and only fill the named gaps.** | True bi-directionality without changing the deepagents `task` tool surface. The parent message becomes: `"prior_call_id=abc123. Fill these missing fields: pe_forward_2026, ev_ebitda_forward_2026"`. | **Cost**: requires a small convention in the subagent prompt; one extra read of `/scratch/`. Net: one extra tool call instead of a full re-boot. |
| 5C. Make subagents **resumable threads** in LangGraph (one thread_id per subagent per parent run). Parent re-uses the thread_id, the subagent's checkpoint resumes. | Most native; matches LangGraph's thread model. **Cost**: requires custom subagent runnable wrapper since `_build_task_tool` does not currently pass a stable per-subagent thread_id. Larger code change. |
| 5D. Replace subagents with shared state: every subagent writes to a structured `data_findings` field in agent state; parent reads from state directly without re-issuing. | Eliminates the tool-message marshalling. **Cost**: requires custom AgentState subclass and reducer (`operator.or_`-style). Works only if subagents are graph nodes, not `task` invocations. |
| **5E.** Combine 5A + 5B: structured contract + conversational refinement on a stable call_id. | **Recommended.** Each subagent (a) writes structured JSON, (b) caches structured JSON keyed by its `task_call_id`, (c) when re-invoked with `prior_call_id=…`, reads the prior JSON and emits a delta. |

### Recommendation: **5E**

Schema (structured response on every subagent):

```python
class CollectionFindings(BaseModel):
    requested: list[str]            # fields the parent asked for
    obtained: dict[str, Any]        # fields successfully collected, with values
    gaps: list[Gap]                 # field name + reason (no_data / provider_unavailable / quota / not_attempted)
    tools_used: list[ToolCallSummary]  # for parent observability
    notes: str | None = None
    call_id: str                    # echoed back; key for /scratch cache
```

Parent prompt rules (in `stock_evaluation.jinja`):

> When a subagent returns `gaps`, decide:
> 1. If `gap.reason == "no_data"` → mark dimension `data_unavailable` and continue. Do NOT re-call.
> 2. If `gap.reason == "not_attempted"` → re-call the *same subagent* with `prior_call_id=<that call_id>` and a description naming only the missing fields.
> 3. If `gap.reason == "provider_unavailable"` → consult the `## Tool lessons learned` block; if no remaining provider option, mark `data_unavailable`.

Combined with Problem 1's lesson injection, the parent never re-tries something that the lessons already say is impossible.

---

## Problem 6 — Run-time budgets and graceful degradation (R3, R7)

The trace burns 42 min because nothing caps it. Even after 30 LLM calls the system has no signal that it should stop.

### Options

| Option | What it does | Tradeoffs |
|---|---|---|
| **6A. Compose `ModelCallLimitMiddleware(run_limit=20)` and `ToolCallLimitMiddleware(run_limit=40)`.** | Native, one-line config. `exit_behavior="end"` produces a clean summary message rather than throwing. | **Cost**: chosen ceiling has to be tuned per agent; too tight = false stops. |
| 6B. Subagent-level budgets via the same middleware on subagent runnables. | Same primitive, applied at a tighter scope. | Recommended in addition to 6A. |
| 6C. Wall-clock budget via `asyncio.wait_for` around `ainvoke`. | Hard timeout. **Cost**: graceless — kills any in-flight tool. |
| 6D. Reflect-and-stop: when the agent's lessons block grows, the orchestrator prompt includes "if you have ≥3 unrecoverable gaps, stop and produce a partial report". | Cheapest. **Cost**: depends on the LLM following the rule. With a stronger model (Problem 2D), reliable. |

### Recommendation: **6A + 6B + 6D**

The middleware caps catch runaway loops; the prompt rule encourages the model to surrender gracefully before the cap. A weaker model needs the hard cap; a stronger model rarely hits it.

---

## Problem 7 — Parallelism (R7)

Trace shows 10 sequential `task` calls. Most have no inter-dependency.

### Options

| Option | What it does | Tradeoffs |
|---|---|---|
| 7A. Rely on the deep-agent's existing parallel-tool-calling capability. The orchestrator prompt (`TASK_SYSTEM_PROMPT` upstream) already encourages parallel `task` calls. With Problem 4C in place, half of the legacy subagents become single tool calls; the LLM is much more likely to issue them in one batch. | **Recommended.** No new code. |
| 7B. Add an explicit "planning" graph node before the deep-agent that emits a list of independent tasks; a "fanout" node dispatches them via `Send`. | Best raw latency for screening/batch flows. **Cost**: less LLM-driven; harder to do for stock_evaluation where the work-set is discovered. |
| 7C. Ban serial tool calls below a certain prompt-rule unless the LLM justifies it. | Prompt-only. **Cost**: only as good as the model. |

### Recommendation: **7A primarily; 7B for `equity_screening` (already partially uses Send) and the `data_collection` phase of `criterion_evaluation`.**

---

## Problem 8 — Observability of adaptation (cross-cutting)

If the system "learns from past experience", we need to see it learning.

### Recommendation

- Every lesson written by `ToolKnowledgeMiddleware` is tagged in LangSmith metadata (`lesson_count`, `lessons_applied_this_call`).
- Add a custom LangSmith run tag `tool_failure_pattern=<error_class>` on every errored `ToolMessage`.
- Add a thin `langsmith trace list --filter "and(eq(metadata.lesson_count, 0), error)"` recipe to spot agents that didn't learn.

---

## Putting it together

The runtime architecture for `stock_evaluation` after the changes:

1. **Orchestrator** (paid Sonnet 4.6 with Haiku fallback) holds:
   - All light/atomic OpenBB tools directly (≈40 tools).
   - `LLMToolSelectorMiddleware(model=haiku, max_tools=8, always_include=["task"])` — filters per turn.
   - `ToolKnowledgeMiddleware` — injects lessons block into system prompt.
   - `ToolRetryMiddleware(max_retries=2, on_failure=lesson_formatter)` — retry transient tool errors with paraphrased error.
   - `ContextEditingMiddleware(ClearToolUsesEdit(trigger=40_000, keep=4))` — auto-trims old tool messages.
   - `SummarizationMiddleware(trigger=80_000)` — fallback if context still grows.
   - `ModelRetryMiddleware(max_retries=3)` + `ModelFallbackMiddleware("haiku-4-5", "openrouter/nemotron-free")`.
   - `ModelCallLimitMiddleware(run_limit=20)` + `ToolCallLimitMiddleware(run_limit=40)`.
   - `task` tool that targets only the 5 *heavy* surviving subagents.
2. **Surviving subagents** (`equity-fundamentals`, `web-search`, `regulatory-filings`, `news`, `data-validation`) each:
   - Use the cheap/fast model (Haiku) since they're tool-routers.
   - Same retry/fallback/knowledge middleware stack.
   - `response_format=AutoStrategy(schema=CollectionFindings)`.
   - Cache structured response to `/scratch/subagent_runs/{call_id}.json`.
   - Read `prior_call_id` from description, load prior JSON, emit delta on refinement calls.
3. **Shared store** carries `("tool_lessons", tool_name)` namespace alongside the existing `("cache", tool_name)` namespace.

Estimated impact on the AMZN trace, holding the model fixed:
- **Lessons**: the 12 forward-sales failures collapse to 1 (lesson "no estimates for AMZN, skip"); the 4 `limit=10` failures collapse to 1 (lesson "≤5"); the 5 `intrinio` calls collapse to 1 (lesson "no key"). Saves ≥20 LLM round-trips.
- **Retries / fallback**: the 5 streaming `APIError`s no longer cost 600 s of dead time.
- **Subagent flattening**: `equity-price`, `equity-estimates`, `economy-macro` were 3 task spawns (~20 min) → become 3 direct tool calls (~30 s with retry).
- **Context**: 30 KB cashflow JSON gets summarised into a 200-token row description; subsequent calls are 30× cheaper.

Combined with switching the default model to a paid one (Problem 2), expected wall time drops from 43 min to **3–6 min** with no per-integration code.

---

## Sequencing

| Wave | Items | Time to ship | Risk |
|------|-------|--------------|------|
| **1 — config & verify** | 2D model defaults; verify `ModelRetryMiddleware` actually fires (failing test → fix); raise `max_retries`; add `ModelFallbackMiddleware`. | 1 day | Low. |
| **2 — context & limits** | 3A (`ContextEditingMiddleware`); 3C (`ToolResultCacheMiddleware` summary-on-write); 6A+6B (call-limit middleware). | 1–2 days | Low — all native. |
| **3 — adaptive learning** | 1E (`ToolKnowledgeMiddleware` + parsers); LangSmith metadata tagging. | 2–3 days | Medium — first system that mutates the prompt at runtime. Land behind a feature flag at first. |
| **4 — architecture** | 4C (flatten 10 of 14 subagents; introduce role-based `LLMToolSelectorMiddleware`); 5E (structured findings + conversational refinement). | 4–6 days | Medium-high — touches 14 files. Land per-domain. |
| **5 — graph parallelism** | 7B for screening / criterion_evaluation. | 2 days | Low. |

Each wave is independently shippable and observable. After Wave 2 the system is already much more reliable; Wave 3 is where it starts *learning*.

---

## Critical files / functions

- [src/muffin_agent/utils/agent_builder.py:475-529](src/muffin_agent/utils/agent_builder.py#L475-L529) — middleware assembly. Verify `ModelRetryMiddleware` placement; add `ModelFallbackMiddleware`, `ContextEditingMiddleware`, `LLMToolSelectorMiddleware`, `ToolKnowledgeMiddleware`, `ModelCallLimitMiddleware`, `ToolCallLimitMiddleware` as new `with_*` builder methods.
- [src/muffin_agent/middlewares/tool_error_handler/middleware.py](src/muffin_agent/middlewares/tool_error_handler/middleware.py) — replace with `ToolKnowledgeMiddleware`. Keep duplicate-blocking as one of its facets.
- [src/muffin_agent/middlewares/tool_result_cache/middleware.py:140](src/muffin_agent/middlewares/tool_result_cache/middleware.py#L140) — add summary-on-write for entries > 20 K tokens; teach LLM via existing `tool_result_cache.jinja` partial.
- [src/muffin_agent/model_config.py](src/muffin_agent/model_config.py) — add `role`, `fallback_models`, drop `:free` default.
- [src/muffin_agent/agents/subagents.py](src/muffin_agent/agents/subagents.py) — collapse from 14 → 5 surviving subagents; export `light_tools(config)` for the orchestrator.
- [src/muffin_agent/agents/data_collection/](src/muffin_agent/agents/data_collection/) — keep only `equity_fundamentals.py`, `web_search.py`, `regulatory_filings.py`, `news.py`. Move tool registration of the others into a new `light_tools.py` consumed by the orchestrator.
- [src/muffin_agent/agents/data_collection/equity_fundamentals.py](src/muffin_agent/agents/data_collection/equity_fundamentals.py) and the other 4 surviving subagents — add `response_format=CollectionFindings`, prompt rules for `prior_call_id` refinement.
- [src/muffin_agent/agents/stock_evaluation.py](src/muffin_agent/agents/stock_evaluation.py) — register flat tools, register selector middleware, register lessons middleware.
- [src/muffin_agent/prompts/stock_evaluation.jinja](src/muffin_agent/prompts/stock_evaluation.jinja) — add the `## Tool lessons learned` placeholder; teach `prior_call_id` refinement protocol.

## Verification

- **Unit:** synthetic mid-stream `APIError` retried (Wave 1); 5 known error families parsed by `ToolKnowledgeMiddleware` parsers (Wave 3); `LLMToolSelectorMiddleware` selects `equity_price_quote` for "what is AMZN trading at?" (Wave 4); subagent emits `CollectionFindings`, parent issues delta on `prior_call_id` (Wave 4).
- **Integration:** force `equity_fundamental_balance` to 422 with `limit=10`; assert next LLM turn sees the lessons block; assert the model now sends `limit=5`. Repeat with `intrinio` missing-key error.
- **End-to-end:** re-run AMZN stock_evaluation after each wave; record wall-time, total LLM tokens, tool-failure count, lesson count. Expect monotonic improvement.
- **LangSmith:** filter `--filter 'and(gte(metadata.lesson_count, 1), eq(error, false))'` should match successful runs that *learned*.

## Open questions

1. **Primary paid model** — Sonnet 4.6 vs GPT-4.1 vs paid Nemotron? Picks fallback chain.
2. **Subagent-flattening line** — accept the "5 surviving subagents" set above, or pick a different cut?
3. **Lesson scope** — lessons live for the *thread* (default), or persist across threads via a shared external store (Redis/Postgres)? The latter makes the system smarter every run but introduces cross-user contamination concerns; namespace by `user_id` is straightforward.
4. **Wave bundling** — do you want Waves 1–2 as a quick safety patch first, or commit to the full Wave-3+ adaptive build now?
