# 📅 Roadmap

## Phase 1

### Data Collection Agents
- [x] Create example data collection agent
- [x] Develop CLI for agents
- [x] Add setup guide including guide on getting API keys for OpenBB providers, setting up langfuse and getting other .env variables
- [x] Create other data collection agents from [docs/data-collection-agents.md](docs/data-collection-agents.md)
    - [x] 1. Equity Fundamentals
    - [x] 2. Equity Price
    - [x] 3. Equity Estimates
    - [x] 4. Equity Ownership & Short Interest
    - [x] 5. Company News
    - [x] 6. Options
    - [x] 7. Economy & Macro
    - [x] 8. Fixed Income & Rates
    - [x] 9. ETF & Index
    - [x] 10. Discovery & Screening
    - [x] 11. Currency & Commodities
    - [x] 12. Regulatory & Filings
    - [x] 13. Fama-French
- [ ] Validate that all MCP tools (except Utility Tools) are assigned to agents
- [x] Handle rate limiting with `openai/gpt-oss-120b:free` model


### Stock Evaluation Agent (v1)
- [x] Developed deep agent that uses data collection agents as sub agents and perform:
    - planning;
    - data collect using sub agents;
    - data validation;
    - analyzis;
    - reflect on results
- [x] Add logic for data validation that checks if data is sufficient, data is relevant, if point of time is provided - data is not going beyond that point of time


### DX
- [x] Create prompt generation skill.
- [x] Extend docker compose to allow mounting local code to the code location within docker to allow making changes locally and test them immediately. *Achieved out-of-the-box: the agent server runs on the host under `langgraph dev` and reads source directly from the local filesystem (native hot reload). Docker holds only infra + chat UI. See [docs/debugging-locally.md](docs/debugging-locally.md).*
- [x] Add configuration to allow debugging python code when docker compose is launched. Options (In order of preference):
    - VS Code Dev containers (mount code directory to sync changes)
    - Using VSCode debugging in docker (attach to running container, mount code directory to sync changes)
    - Running muffin agent server outside of docker compose with vscode debuger (having separate docker compose file for development that uses server hosted outside)
    - Running muffin agent server outside of docker compose with `langgraph dev` (or alternative) and attach to running process with vscode (having separate docker compose file for development that uses server hosted outside)
    -  Using normal VSCode python debugger to execute muffin-cli, just connect it to other services hosted within docker compose (least preferred)

    *Implemented via option 4 (`LangGraph Dev Server (Debug)` launch config; `docker-compose.dev.yml` overlay hides `langgraph-api` behind a `production` profile so host port 8123 is free for `langgraph dev`, publishes MCP host ports, and swaps the OpenSandbox config for host-reachable sandbox endpoints) and option 5 (per-agent CLI debug — 26 existing launch configs). Options 1-2 remain as future enhancements for fully in-container development. See [docs/debugging-locally.md](docs/debugging-locally.md).*

### Documentation
- [ ] Document Data Validation agent and add launch.json config
- [x] Document Criterion Evaluation agent and add launch.json config
- [x] Document Market Regime Agent (README.md, CLAUDE.md, roadmap.md)

### Sandbox
- [x] Setup the model to generate python code for deterministic functions instead of doing math on it's own.
- [x] Update subagents to re-use existing backend or kill backend after each tool execution
    - `SandboxFactory` discovers sandboxes by `thread_id` metadata, creates if not found
    - `get_backend` (deep agent) and `execute_python` (subagent tool) reuse the same container
    - Dead containers are auto-recreated transparently (in-sandbox state is lost)
- [ ] Rework `execute_python` to generic `execute`
- [ ] Update prompts to guide agents to execute code for computations instead of computing within LLM call

### Deployment
#### Option 1 (Separate client and agent server):
##### For Server options:
- [x] Setup self-hosted [Standalone Agent Server](https://docs.langchain.com/langsmith/deploy-standalone-server#docker-compose) accept 1M node executions limit for development purpose. Build image with [langgraph cli](https://docs.langchain.com/langsmith/cli#build)
- [x] Docker Compose: all 13 infrastructure services start healthy (`firecrawl-api` requires `--start-docker` flag + dedicated `firecrawl-postgres` using `ghcr.io/firecrawl/nuq-postgres:latest` with `NUQ_DATABASE_URL` set; healthcheck uses root `/` endpoint not `/health`)
- [x] `langgraph-api` crashes on startup: `ImportError: attempted relative import with no known parent package` — graph.py loaded as script instead of package module; needs investigation
- [ ] ~~Setup [aegra](https://github.com/ibbybuilds/aegra)~~
##### For client:
- [x] Setup client web app. For MVP we can go with [langchain-ai/agent-chat-ui](https://docs.langchain.com/oss/python/langchain/ui)
- [ ] ~~Use [LangSmith studio](https://docs.langchain.com/langsmith/studio)~~
- [ ] ~~Use [Agent Chat UI](https://agentchat.vercel.app/)~~
#### Option 2:
- [ ] ~~Go with [chainlit](https://docs.chainlit.io/integrations/langchain) for both client and server~~
#### For both options:
- [x] Make sure that integration with langfuse still works. The CLI attaches the handler per-run via `RunnableConfig(callbacks=...)`, but the deployed graphs are invoked by the LangGraph Platform server, which never saw those callbacks — so tracing worked from the CLI but not in deployment. Fixed by `utils/observability.py:instrument_graph(graph)` (LangFuse's documented LangGraph Server pattern: bake the handler in via `graph.with_config({"callbacks": [CallbackHandler()]})`); wired into all five `langgraph.json` entrypoints. (`session_id` grouping still pending — see the `propagate_attributes()` item below.)

## Phase 2

### Data Validation Agent
- [x] Develop data validation agent that takes criterion and data collected and checks if data is sufficient, data is relevant, if point of time is provided - data is not going beyond that point of time. Add this agent as sub agent to Stock Evaluation Agent. Agent should produce confidence/relevance scores.

### Criterion evaluation Agent
- [x] Develop deep agent that takes criterion that needs to be evaluated and with that criterion:
    - defines data needs;
    - calls data collection subagents to collect this data;
    - calls data validation agent to validate the data;
    - evaluates criterion using the data (produces confidence/score/reasoning);
    - reflects on evaluation results and based on reflection results push back on analysis to gather more data or re-evaluate if needed.

### Criteria Definition Agent
- [x] Develop criteria definition agent that classifies a ticker by sector, market type (DM/EM), and stock type (value/growth), then loads matching valuation skills (55 SKILL.md files across 10 sectors) to produce sector-specific evaluation criteria with target ranges. Standalone deep agent — not yet wired into the investment pipeline.
- [x] Build `SkillSuggestionMiddleware` — metadata-based skill filtering via `get_suggested_skills` tool. Replaces listing all 55 skills in the system prompt with a compact summary + tool-based discovery. Added `metadata` tags (sector, market, stock_type, sub_sector, scope) to all 55 SKILL.md frontmatter. Reusable for any agent with tagged skills.
- [ ] Add valuation skills for remaining sectors: Utilities, Materials/Mining, Conglomerates, Healthcare Equipment, Semiconductors, Fintech/Payments, Aerospace & Defence, Transportation & Logistics
- [ ] Wire criteria definition agent into the investment pipeline (as Stage 0 or parallel with Group 1)

### Criteria evaluation Agent
- [x] Develop agent that takes as an input ticker and some additional information and: classifies the ticker, defines list of criteria, calls a criterion evaluation subagent per criterion, and synthesises a final verdict. Shipped as `agents/criteria_analysis/` — a LangGraph orchestrator with 5 stages:
    - Stage 1 `ticker_classification_node` (3 data subagents + validation) producing `TickerClassificationOutput`; short-circuits when CLI flags pre-supply classification.
    - Stage 2 `criteria_definition_node` (skill-filtered, in parallel with Stage 3) — wraps the existing `criteria_definition` agent.
    - Stage 3 `valuation_methodology_node` (web-search + discovery-screening subagents) — surfaces ticker-specific criteria the skills miss.
    - Stage 4a `merge_criteria_node` — deterministic Python dedup with canonical-name matching, weight renormalisation, source tagging.
    - Stage 4b `criterion_evaluation_node` — `Send` fan-out, one per merged criterion; `criterion_evaluation` agent upgraded to emit `CriterionEvaluationOutput` via `AutoStrategy`.
    - Stage 5 `synthesis_node` — reasoning-only deep agent producing `CriteriaAnalysisSynthesis` (composite score, signal, weighted breakdown, positives/negatives, divergences, thesis paragraph).
    
    CLI: `muffin criteria-analyze TICKER`. Registered in `langgraph.json` as `criteria_analysis`. Reflection-loop pushback (synthesis re-running with new criteria) deferred — see follow-ups below.

#### Criteria analysis follow-ups (deferred)
- [x] **Native compiled-agent-as-node refactor (2026-07)** — replaced the `_node_helpers.invoke_structured_agent` wrapper pattern (which invoked deep agents with a silently-ignored `{"input": ...}` key → the model saw zero messages and hallucinated classification/reasoning) with the trading-analyst / council-persona composition: each stage is a compiled agent added via `add_node(..., input_schema=<TypedDict>)`, task context rendered into the system prompt (`with_runtime_system_prompt_template`), structured response unpacked via single-field wrapper output models. Also: `make_graph` config-only factory, `RetryPolicy(max_attempts=2)` + fail-loud (no fallback dicts), `criterion_evaluation` as a 2-node worker subgraph, E2E test now in `COVERED_GRAPHS`. Cross-cutting fixes shipped alongside: `append_block`/`_RuntimePromptMiddleware` content-blocks preservation, `investment/utils.py` point-fix, `tool_lessons_mode` config knob.
- [ ] **Migrate the investment pipeline (`agents/investment/`) off `run_deep_agent_node`** to the same native compiled-agent-as-node composition (drop the wrapper + fallback-dict pattern). The `{"messages": [...]}` + `config` point-fix already landed; the full refactor mirrors what criteria_analysis now does.
- [x] **Tool telemetry (2026-07)** — per-tool execution records (`tool_runs`) via a reducer-backed state channel; the muffin-ui criteria page renders a per-criterion "Data collection" timeline + a run-level tool-execution summary. Records are reconstructed from message history in `aafter_agent`.
- [x] **Capture middlewares consolidated → `agent_capture/` (2026-07)** — `subagent_transcript` + `tool_telemetry` merged into one middleware pair doing a single message-walk (shared full-fidelity serializer keeping `status` + cache-hit; transcript channel + tool-records channel; builder excludes the response-format schema name from records). Also fixed the prod telemetry no-op: the `tool_telemetry_enabled` gate read the ambient `get_config()` inside `aafter_agent`, which silently returned nothing on the deployed LangGraph runtime while local tests passed — capture is now unconditional and graphs opt in by declaring the `tool_runs` channel on their state.
- [ ] **Deep-agent-inside-deep-agent transcripts — verify the runtime guard on the deployed server before nesting deep agents.** A deep agent captures its transcript (`subagent_runs`) only when it was invoked by another agent's `task` tool, detected at runtime via `configurable.ls_agent_type == "subagent"` read through `get_config()` in `aafter_agent`.
    - **Works everywhere:** ReAct subagents always leave transcripts; a top-level deep agent correctly skips capture (its transcript would duplicate the thread's own `messages`); `tool_runs` capture is unconditional and does not use this guard.
    - **Works locally, unverified on the server:** a deep agent nested as a `CompiledSubAgent` inside another deep agent leaves a transcript (locked by `test_guarded_capture_fires_for_deep_agent_as_subagent`). The risk: the guard's `get_config()` call is the same mechanism that silently disabled tool telemetry on the deployed server. If it fails there, the guard returns False and the nested deep agent's transcript is **silently skipped** — fail-closed (never duplicates, just missing).
    - **Currently harmless:** no deployed graph nests deep agents via `task` (all production subagents are ReAct collectors).
    - **Action:** before shipping any deep-in-deep composition, verify the guard fires on the server (one nested-deep test run), or preemptively replace the detection with a server-proven mechanism — e.g. stamp a state flag from `awrap_tool_call`, where `request.runtime.config` is known to work in prod (the tool-result cache reads it there).
- [ ] **Tool telemetry — extend + enrich.** Wire `tool_runs` rendering for the other graphs (`research` / `trading_decision` / `stock_evaluation` / `investment`) — the middleware is already universal, they just need `tool_runs` on their graph state + the UI on their screens. Enrichments: per-tool **duration** (needs per-call timing, e.g. `awrap_tool_call` writing timestamped records rather than the current message-walk); live per-tool row streaming via a custom stream mode / `get_stream_writer()` (records currently land when each node finishes); itemising individual `ToolRetryMiddleware` attempts rather than showing only the final outcome.
- [x] **Data-source truthing + per-criterion custom stream events (2026-07)** — prod runs on a free OpenRouter route (`nvidia/nemotron-3-super-120b-a12b:free`) revealed the worst failure mode: every stage agent **single-shot the structured output with zero tool/`task` calls and fabricated `data_sources`/evidence** (verified via Langfuse: 0 TOOL spans across entire criteria runs; the M11 telemetry UI was correctly showing "nothing"). Shipped: (a) `_reconcile_data_sources` in `package_evaluation_node` — deterministic anti-hallucination pass that strips `data_sources` and caps `confidence` at 0.3 when no tools ran (`data_collected: false` flag for the UI badge), and drops uncorroborated source names when tools did run; (b) `criterion_evaluated` custom stream event via `get_stream_writer()` from the package node — the only live per-criterion signal, since ALL parallel Send workers commit in ONE parent superstep (clients need `stream_mode=["custom",…]` **and** `subgraphs=True`; without subgraphs, nested custom events are NOT delivered — verified against langgraph 1.2.8); (c) non-negotiable data-collection contracts added to the criterion-evaluation / classification / criteria-definition / valuation-methodology prompts.
- [ ] **Data-collection enforcement loop (middleware)** — model-agnostic backstop for free/weak models that ignore the prompt contract: an `aafter_agent` (or response-format-wrapping) guard that, when the agent produces its structured response with zero non-plumbing tool/`task` records, injects a corrective feedback message ("you must collect data via your subagents before scoring") and re-enters the loop, bounded to 1–2 retries; after exhaustion, accept the response with `data_collected=False` so the deterministic truthing (above) still tells the truth downstream. Generalise beyond criteria to every deep-agent stage (classification / definition / methodology / stock_evaluation).
- [ ] **Stage-level custom-event progress protocol** — extend the shipped per-criterion `criterion_evaluated` event with per-stage *payload* events where the UI needs detail beyond structure. **Scope narrowed (2026-07):** muffin-ui migrated the run views to the `@langchain/react` protocol-v2 `useStream`, whose `subgraphsByNode` discovery already delivers live per-node **structure + status** (`running`/`complete`/`error`) for every compiled-agent stage (criteria stages/workers, council personas, trading analysts) with no backend events and no node-name regexes. So the remaining need is **payload**, not lifecycle: emit a custom event only where a stage's headline result should stream ahead of its subgraph committing (the criteria worker already does this via `criterion_evaluated`). One place this still matters: council persona INNER-node progress (collect→compute→verdict) is depth-2 and the UI's root pump is depth-1, so today it's inferred from the persona's depth-1 values events — a per-persona `stage_progress` custom event (or a depth-2 lifecycle projection on the SDK side) would make it exact. Cross-ref: muffin-ui ROADMAP "M12 follow-ups".
- [ ] **Structured-output range validators** — a live run recorded `confidence: 95.0` where the schema documents `0.0–1.0`; add Pydantic `Field(ge=0, le=1)` (or validators) on the score/confidence fields across `criteria_analysis` / `criterion_evaluation` schemas so out-of-range LLM output is rejected/clamped instead of silently stored.
- [ ] **Reflection-loop pushback** — after synthesis, run a reflect step that can either re-issue specific criterion evaluations with refinement notes (via `SubagentRefinementMiddleware`'s `prior_call_id` protocol) or request additional criteria from a re-run of `valuation_methodology_node` with the synthesis as input context.
- [ ] **Per-criterion concurrency cap** — currently uncapped (`equity_screening` parity). If OpenBB/MCP load becomes an issue, cap via a semaphore inside `_fan_out_criteria` or split the fan-out into batched waves.
- [ ] **LLM merge reconciliation pass** — when more than N (default 8) criteria survive deterministic dedup, optionally invoke `get_summariser()` to rank and trim. Hidden behind a `merge_with_llm: bool = False` flag.
- [ ] **`target_price` in synthesis** — add an optional `target_price: float | None` to `CriteriaAnalysisSynthesis` if downstream consumers need a price anchor without running the separate `valuation_node` from `investment_analysis`.
- [ ] **Cross-graph composition** — make `criteria_analysis` invokable as a sub-graph of `investment_analysis` (replacing or augmenting `thesis_synthesis_node`) so investors get both the criterion-weighted view and the DCF/multiples view in one run.

## Core workflow

### High-level
1. [ ] Define screening zone
    - [ ] Analyse Region against other regions
    - [ ] Analyse Country (Economy) against other economies
    - [ ] Analyse Sector against other sectors
2. [ ] Define Ticker
    - [ ] Check losers/gainers/news
3. [ ] Analyse Ticker
    - [ ] Analyse business. Idea, Technologies used and their potential, MOAT and competetive advantage, offered products, current market coverage and potential market coverage, expansion to other markets (requires understanding of economy of potential markets, should be computed by subagent and stored in store)
    - [ ] News analysis. Defining if news are affecting business short-term or long-term and how.
    - [ ] Fundamental analysis
    - [ ] Technical analysis
    - [ ] Forecasting and scenario modeling
    - [ ] Risk assesment
    - [ ] Valuation
4. [ ] Compare against other tickers (requires full analysis of other tickers, should be computed by subagent and stored in store)
5. [ ] Define trade proposal 
6. [ ] Check how new investment fits to portfolio
7. [ ] Execute the trade

### Low-level
- [ ] Idea Sourcing & Screening: Defines investment idea (Step 1 from [docs/investment-process.md](docs/investment-process.md))
    - [ ] Macro screeners:
        - [ ] Sector screener: Compare sectors in the current economic condition to define which has potential to attract more capital.
        - [ ] Country screener: Compare countries in the current economic condition to define which has potential to attract more capital.
        - [ ] Technology screener: Search and compare cutting edge technologies, latest advancmenets to define which has potential to attract more capital and/or has high potential to attract many customers and get good market share.
    - [ ] Ticker Screeners:
        - [ ] Loser screener: Check weekly/daily losers to later define if companies fairly lost capitalization or if it's temporary (not reasonable long-term) lose. This screener should be able to analyze in specific country/sector/market cap.
        - [ ] Gainers screener: Check weekly/daily losers to later define if companies fairly gained capitalization or if it's temporary (not reasonable long-term) gain. This screener should be able to analyze in specific country/sector/market cap.
        - [ ] News screener: Check news to define which companies require attention.
    - **TODO**
- [ ] Idea Evaluation (Steps 2-4 from [docs/investment-process.md](docs/investment-process.md))
    - [x] Step 2 — Market Regime Agent: classifies macro/liquidity regime across 4 dimensions (growth, inflation, monetary policy, liquidity/risk appetite); produces factor tilts and positioning guidance; supports ticker / sector+industry+country / query-only context modes; structured output via `AutoStrategy(schema=MarketRegimeOutput)`.
    - [x] Step 3 — Sector / Industry Agent: assesses sector/industry attractiveness across 6 dimensions (cycle position, Porter's Five Forces competitive structure, thematic drivers, relative valuation, regulatory backdrop, alpha opportunity/dispersion); 4 data subagents (etf-index, discovery-screening, news, regulatory-filings) + data-validation; supports ticker / sector+industry / query-only context modes; structured output via `AutoStrategy(schema=SectorViewOutput)`.
    - [ ] Step 4 — Business, Moat, Management & ESG Triage Agent
- [ ] Ticker Valuation and forecasting (Steps 5-6 from [docs/investment-process.md](docs/investment-process.md))
    - Do valuations based on fundamentals
    - [x] Step 6 — Forecasting & Scenario Modeling Agent: builds 3-year bull/base/bear forward model anchored to analyst consensus; 4 data subagents (equity-estimates, equity-fundamentals, economy-macro, currency-commodities) + data-validation; sandbox computes historical calibration, scenario projections, sensitivity table, accruals ratio; probability anchors keyed to company_signal (pass=60/25/15, watch=50/25/25, fail=40/25/35); structured output via `AutoStrategy(schema=ForecastOutput)`.
    - [x] Step 7 — Valuation & Relative Value Agent: computes intrinsic value via blended DCF (exit-multiple + Gordon Growth), EV/EBITDA/P/E/FCF-yield multiples, and scenario-weighted NAV; 4 deterministic tools (compute_wacc, compute_dcf, compute_multiples_value, compute_scenario_weighted_value); 5-year own-history relative value vs. peer_median and market_median; 5 data subagents (equity-price, equity-estimates, etf-index, discovery-screening, fixed-income) + data-validation; valuation_signal (cheap/fairly_valued/expensive); runs sequentially after Group 2 barrier; output consumed by thesis_synthesis_node.
- **TODO**
- [ ] Analysis check (Steps 8-9 from [docs/investment-process.md](docs/investment-process.md))
    - [x] Step 8 — Risk & Downside / Stress Testing Agent: quantifies idiosyncratic and systematic risk via 4 deterministic tools (compute_beta, compute_var_cvar, compute_sharpe_sortino, compute_max_drawdown); FF5+UMD 6-factor OLS regression via sandbox; IV term structure (30/60/90d + 25d skew); short interest crowding classification; 6 stress scenarios (2 fixed historical: GFC 2008, COVID 2020; 3 regime-derived; 1 idiosyncratic); ex-ante stop level; risk_signal (acceptable/elevated/unacceptable); 7 data subagents (equity-price, options, fama-french, equity-ownership, fixed-income, economy-macro, data-validation).

## Phase 3

### Agent Evaluations
- [ ] Add support of defining point of time at which data has to be fetched

### Data collection
- [ ] Iterate over data collection agents, improve prompts based on openbb docs. if needed split to smaller specialized agents.
- [ ] Check https://docs.openbb.co/odp/python/reference . There are a lot of commands that are not covered by MCP.
- [x] Add firecrawl to collect data from web (consider adding as MCP)

### Specialized Agents
- [ ] Integrate tool(s) to get Technical indicators (consider TA-lib)
- [ ] Develop Specialized Technical Analysis Agent
- [ ] Develop Specialized Fundamental Analysis Agent
- [ ] Develop Specialized Macro economy Analysis Agent
- [ ] Develop Specialized News & Sentiment Agent
- [ ] Develop Specialized Social Networks Agent
- [ ] Develop Specialized Prediction Market Analysis Agent
- [ ] Develop Specialized Strategic & Growth Agent
- [ ] Develop Specialized Competitive Analysis Agent
- [ ] Explore agents from https://github.com/virattt/ai-hedge-fund

### TradingAgents port — `agents/trading_decision/`

Composable building blocks from [TauricResearch/TradingAgents](https://github.com/TauricResearch/TradingAgents). Designed as a standalone module that accepts a generic `AnalysisContext`, decoupled from muffin's `investment_analysis` pipeline.

- [x] **PR 1 — Investment debate spine.** Bull Researcher + Bear Researcher + Investment Judge synthesis. `build_investment_debate_graph` + `AnalysisContext` envelope + `InvestmentJudgeOutput` schema + speaker-tag debate routing (default 2 rounds = 4 turns, tunable via `configurable.max_investment_debate_rounds`).
- [x] **PR 2 — Trader agent.** `TraderOutput` (3-tier action / entry_price / stop_loss / take_profit / position_sizing / time_horizon / reasoning). Translates the Judge's signal into operational instructions. Exposes `build_investment_thesis_graph` (debate → Judge → Trader). Conviction-bucketed default sizing table baked into the prompt; trader node skips the LLM when Judge output is missing or errored to avoid compounding noise.
- [x] **PR 3 — Risk debate + Portfolio Manager.** Aggressive / Conservative / Neutral round-robin (default 1 round = 3 turns, tunable via `configurable.max_risk_debate_rounds`) → Portfolio Manager producing canonical `PortfolioDecisionOutput` (5-tier rating, price target, time horizon, position sizing, key risks remaining, confidence + `incorporates_past_lessons` PR 4 flag). Exposes `build_trading_decision_graph` (the full pipeline). Speaker-tag routing via explicit `latest_speaker` enum (cheaper than parsing three tag prefixes).
- [x] **PR 4 — CLI command + reflection memory.** `muffin decide <TICKER> --narrative "..."` ships. Outcome-driven reflection: per-user `BaseStore` namespace `("memories", user_id, "decisions")`, OpenBB-fetched realised returns + alpha vs SPY (tunable benchmark), Reflector LLM produces 2–4 sentence reflection, injected into future Portfolio Manager prompts (up to 5 same-ticker + 3 cross-ticker). Pipeline degrades silently when store/user_id/reflection_enabled are absent. CLI uses `InMemoryStore` so reflection accumulates only within a single Python process — see "persistent store backend" follow-up below.

  **Follow-up — persistent reflection store**: The CLI's `InMemoryStore` does not survive process restart, so reflection memory cannot accumulate across CLI invocations today. Options: (1) ship a file-backed `BaseStore` subclass (SQLite or JSON), (2) document the LangGraph `PostgresStore` recipe in the deployment guide, (3) wait for users to deploy on LangGraph Platform (which injects a persistent store automatically). Pick after first round of real usage.
- [x] **PR 5 — Adapter for investment_analysis.** `AnalysisContext.from_investment_analysis_state(state)` pure adapter + `muffin decide --analysis-json <path|->` flag for two-step CLI composition (upstream graph writes JSON state, `muffin decide` consumes it). `--narrative` is now optional; both flags can be combined to layer structured fields with free-form notes.

  **Follow-up — `--from-analysis` one-shot orchestration**: would run `build_investment_analysis_graph` internally then pipe its state through the adapter. Currently blocked on `thesis_synthesis_node` (raises `NotImplementedError`), so the analysis graph cannot complete in a single invocation. Options when we revisit: (1) implement `thesis_synthesis_node` using `build_trading_decision_graph` (couples the two pipelines), (2) add an `interrupt_before` kwarg to `build_investment_analysis_graph` so the CLI can stop before the stub, (3) catch the exception + recover state via the checkpointer. Also wire a `--json` flag to `muffin analyze` so users can pipe its output to `muffin decide` directly.
- [x] **PR 6 — Docs + composition examples.** README + `docs/trading-decision.md` composition guide + `.vscode/launch.json` debug configs for `muffin decide`. Documents composition patterns (narrative, from `investment_analysis` state, hybrid).
- [x] **Refactor — LangGraph-native primitives.** Replaced per-agent `MuffinAgentBuilder` factories with single per-role node functions that resolve LLMs via `ModelConfiguration` + `with_fallbacks` + `with_retry` (+ `with_structured_output` for structured roles). Flat state with `Annotated[list[str], operator.add]` reducers per speaker (drops sub-state structs, speaker-tag matching, count fields). Routing back at graph level via `add_conditional_edges` (list form). Per-node `RetryPolicy` as the second retry layer. No try/except in LLM nodes — failures propagate. Typed `TradingDecisionConfiguration(BaseConfiguration)` replaces ad-hoc `configurable.get(...)`. Shared Jinja partials (`_analysis_context.jinja`, `_investment_debate_state.jinja`, `_risk_synthesis_inputs.jinja`) pulled via `{% include %}`. Two future migration paths documented (custom subgraph with `ToolNode` vs `MuffinAgentBuilder` agent as graph node).
- [x] **Independence — port the Analyst layer (Market / Fundamentals / News / Social) from TradingAgents.** Dropped the hard dependency on `agents/investment/` outputs (`market_regime`, `sector_view`, `company_analysis`, `forecast`, `risk_assessment`, `valuation`). The package is now self-contained: four compiled analyst ReAct agents run in parallel after `reflector_resolve` and write free-text prose reports into top-level state fields (`market_report`, `fundamentals_report`, `news_report`, `sentiment_report`); downstream Bull/Bear/Judge/Trader/PM nodes consume the reports. Analysts use OpenBB MCP for data (price / fundamentals / news / ownership / `news_world`) + Firecrawl MCP for social, plus one new local `get_indicators` @tool (stockstats over OpenBB OHLCV — fills the technical-indicator gap left by OpenBB MCP). Schema simplification: `AnalysisContext` deleted entirely (and its `from_*` adapters); `TradingDecisionState` now has flat `ticker` / `query` / `narrative` + the 4 report fields. CLI: `--analysis-json` flag removed; the four analysts fetch their own data. Graph builders became async to amortise the analyst-agent construction to graph-build time.
- [x] **MuffinAgentBuilder extensions.** Three additive features that make the "compiled agent as direct parent-graph node" pattern clean (used by the analyst layer): (1) `with_state_schema(schema)` forwards a custom `AgentState` extension to `create_agent`, and combined with `OmitFromSchema(input=…, output=…)` annotations derives the agent's `input_schema` / `output_schema` for parent-graph mapping; (2) `with_runtime_system_prompt_template(template)` registers an internal middleware that re-renders the system prompt from agent state on every model call, introspecting the state schema to pass only input-eligible Jinja vars; (3) auto-trigger structured-response → state unpacking when both `with_state_schema(...)` AND `with_response_format(...)` are set, so each Pydantic field maps to a same-named state field for the parent graph to read by name. Backward compatible (no existing caller sets `state_schema`).
- [x] **Multi-agent conference framework + `risk_debate` migration.** Generic, reusable subgraph builder lives at `src/muffin_agent/multi_agent/`. Four pluggable Protocols: `Participant` (`LLMParticipant` for prompt-text rendering, `LLMMessageParticipant` for messages-thread, `AgentParticipant` for wrapping compiled muffin agents with persistent per-thread state), `Moderator` (`RoundRobinModerator`, `AlternatingModerator`), `Terminator` (`MaxRoundsTerminator`), `Judge` (`StructuredOutputJudge`). `build_conference_graph(...)` generates a `StateGraph` with one named node per participant + a pure-Python `dispatch` node that handles speaker selection and termination. Shared `messages: Annotated[list[BaseMessage], add_messages]` state field is the single source of truth — one name-tagged `AIMessage` per turn. AgentParticipants are added as parent-graph nodes via a `prep → agent → extract` wrapper subgraph; per-agent state persists across turns via LangGraph's per-thread mechanism (caller builds the agent with `MuffinAgentBuilder.with_checkpointer(True)`; the parent must provide a real checkpointer instance). Risk debate (3-way Aggressive/Conservative/Neutral) migrated as the proof point: dropped 3 hand-coded debator node files + `_route_risk_debate` + 3 per-speaker reducer fields; replaced with one `_build_risk_debate_subgraph(max_rounds)` call. Bull/Bear (investment debate) intentionally untouched — see follow-up below.

- [x] **Dependency upgrade pass.** Bumped all direct dependencies to latest stable: `langgraph 1.2.0 → 1.2.1`, `langchain-openai 1.1.10 → 1.2.2`, `langgraph-api 0.7.60 → 0.8.7`, `deepagents 0.6.1 → 0.6.3`, `opensandbox 0.1.5 → 0.1.9`, `langfuse 3.14.5 → 4.6.1` (major — `update_current_trace` API removed; `setup_tracing` no longer wires session_id pending a `propagate_attributes()` context-manager rewrite), `mypy 1.19.1 → 2.1.0` (stricter; added two `type: ignore[type-var]` in `multi_agent/conference.py` for `dict[str, Any]` node-state args). Pinned minimums in `pyproject.toml` for reproducibility.

#### TradingAgents port — follow-ups (deferred)

- [ ] **Investment debate (Bull/Bear) migration onto the conference framework.** The 2-way alternating bull/bear debate still uses bespoke node functions + `_route_investment_debate` + per-speaker reducers. Migration is mechanical: `AlternatingModerator("bull_researcher", "bear_researcher")` + `MaxRoundsTerminator(num_participants=2)` reproduces today's routing. Two open items before the migration: (a) decide whether Investment Judge becomes a `StructuredOutputJudge` inside the conference (one fewer parent-graph node) or stays as a separate post-conference node like the Portfolio Manager (keeps the trace topology familiar); (b) the bull/bear prompts use a bespoke `opposing_last` variable for direct rebuttal — the framework already exposes `last_opposing_message` on `LLMParticipant`, so the templates need a one-line rename.
- [ ] **Langfuse v4 `propagate_attributes()` wiring.** The v3 → v4 upgrade dropped `client.update_current_trace(session_id=...)` in favour of a `propagate_attributes()` context manager that scopes attributes (session_id, user_id, metadata, tags) to a code block. `setup_tracing()` in `utils/observability.py` no longer attaches session_id today; expose a thin per-invocation wrapper (or a CLI-level decorator) that calls `propagate_attributes(session_id=...)` around the agent invocation so tickers and user ids tag traces again.
- [ ] **LLM-driven moderator + consensus-detecting terminator** — the framework's `Moderator` / `Terminator` Protocols are pluggable but only rule-based built-ins ship today. A future "Manager + analysts + debators" conference (use case 3 from the conference design notes) will likely want an LLM moderator that picks the next speaker based on what's been said, and/or a `ConsensusJudge` that runs after each turn to decide early stop. Ship when the first use case lands.
- [ ] **`output_language` config knob** — upstream TradingAgents supports per-run language selection for analyst / PM outputs while keeping internal debate in English (via a `get_language_instruction()` helper threaded into agent prompts). The port is English-only. Add an `output_language: str = "English"` field to `TradingDecisionConfiguration` and wire a conditional language-instruction line into the analyst + Portfolio Manager Jinja templates when the demand surfaces.
- [ ] **Reflection memory rotation** — upstream caps resolved entries via `memory_log_max_entries` to prevent unbounded growth of the markdown log. The port relies on the underlying `BaseStore`'s retention. If `PostgresStore` retention proves insufficient in long-running deployments, add `reflection_max_resolved_entries: int | None = None` to `TradingDecisionConfiguration` and prune oldest resolved entries on `decision_writeback` when exceeded.
- [ ] **Server-side look-ahead-bias cap for backtesting** — upstream filters OpenBB OHLCV / financials by `curr_date` at the dataflow layer (`load_ohlcv` + `filter_financials_by_date`). The port pushes `decision_date` into analyst prompts and trusts the LLM to set the OpenBB tool `end_date` correctly. `get_indicators` is already safe (server-side `curr_date`). For rigorous historical backtesting at arbitrary past dates, wrap the OpenBB MCP tool list in a `decision_date`-aware adapter that hard-caps `end_date` / `start_date` / `as_of_date` arguments before forwarding to MCP. Defer until we actually run a backtest harness.

### Research Agent — follow-ups

The MVP Research Agent (`src/muffin_agent/agents/research/`) ships with web search + scrape only and a Vane-style hybrid pipeline. Follow-ups, in priority order:

- [ ] **Academic source tool** (Semantic Scholar / arXiv) — wrap as a plain `@tool`, expose via `extra_tools=` + `extra_sources=["academic"]`. Drop a universal skill into `/skills/research/sources/academic/`.
- [ ] **News source tool** (NewsAPI wrapper, or Firecrawl search with SearxNG news engines). Similar shape to academic.
- [ ] **Finance source wiring**: package existing data-collection subagents (`news_company`, `equity_fundamentals`, …) as `extra_tools` and document the integration recipe so investment agents can plug research into their workflow.
- [ ] **LangSmith eval dataset + pipeline**: build a benchmark for citation accuracy, source diversity, recall on canonical research queries.
- [ ] **Persistent vector cache + Supabase pgvector** — replaces ephemeral OpenAI embeddings with a cached lookup; also serves as the PostgreSQL host for LangGraph checkpoints, replacing SQLite for multi-user deployment. Adds provider-agnostic embeddings (Voyage AI, Cohere, Nomic) as a side benefit.
- [ ] **Fact-checking verifier node** between writer and END: claim-by-claim re-check against sources; downgrades confidence on uncited claims; flags hallucinated citations.
- [ ] **Source credibility scoring**: domain-reputation list + recency penalty + author/publisher metadata; surfaces in the `Source` model.
- [ ] **Streaming progress UI** (LangGraph Studio integration + agent-chat-ui block-style rendering for a live research panel). Migration trigger for swapping the researcher's deep-agent internals to a multi-node sub-graph.
- [ ] **Multi-modal evidence**: PDF, image, video chunking + embedding.
- [ ] **Discussion source tool** (Reddit, HN, StackExchange) for opinion/sentiment queries.
- [ ] **Wolfram Alpha / calculator integration** for numerical / unit-conversion queries.
- [ ] **Conversation-aware suggestion generation**: replace bundled `suggested_followups` with a separate cheap-LLM call after the answer renders (Vane pattern) for snappier UX.

### Reliability v1 (post-AMZN trace `019dc4bc-…`)

A 42.8-min stock_evaluation run identified seven compounding root causes (slow free model, no retry/fallback, no episodic learning, context bloat, etc). Shipped fixes:

- [x] **W1** — `ChatOpenRouter` integration + per-role configurable model chains in `ModelConfiguration` (`orchestrator_models` / `collector_models` / `reasoner_models`, comma-separated env vars). `get_llm_for_role(role)` returns `(primary, fallbacks)` via `langchain.chat_models.init_chat_model` so cross-provider chains work.
- [x] **W2** — `MuffinAgentBuilder.with_fallback_models(*models)` wires `ModelFallbackMiddleware` outermost. Dropped `:free` from `DEFAULT_MODEL` (free OpenRouter routes are unsafe as production default).
- [x] **W3** — `with_context_editing` / `with_summarization` opt-ins (see Other improvements above).
- [x] **W4** — `ToolResultCacheMiddleware` reworked with strict-content invariant (no prose mixed into tool JSON, cache provenance moved to `additional_kwargs["cache"]`). Size-based offload delegated to deepagents `FilesystemMiddleware._aintercept_large_tool_result` (default 20K-token threshold) — no duplication.
- [x] **W5** — `with_model_call_limit(*, run_limit, thread_limit, exit_behavior)` and `with_tool_call_limit(*, tool_name=None, run_limit, thread_limit, exit_behavior)` plus per-tool inline `run_limit` / `thread_limit` kwargs on `with_tool(...)`.
- [x] **W6** — Universal `ToolRetryMiddleware` for transient tool errors. `_should_retry_tool_call` filter targets `ToolException` messages with 5xx / gateway / connection / timeout substrings; 4xx / validation / missing-credential errors propagate so `ToolKnowledgeMiddleware` can record them as lessons.
- [x] **W7** — `ToolKnowledgeMiddleware` (see lesson-from-failures item above).
- [x] **W8** — `SubagentRefinementMiddleware` + `SubagentRefinementParentMiddleware` provide a generic refinement protocol: subagents emit `CollectionFindings` (typed `gaps` with `reason` + `retry_advice`), findings cache to `/scratch/subagent_runs/<call_id>.json`, parent re-issues with `prior_call_id=<id>` to fill gaps without restarting. Auto-wired by `with_subagent_refinement()` based on whether subagents are wired (parent vs child role).
- [x] **W9** — Parallelism prompt rule + graceful-stop rule (≥3 unrecoverable gaps → stop) added to `stock_evaluation.jinja` and `criterion_evaluation.jinja`.

#### Reliability follow-ups (deferred)

- [ ] **`Send` fanout for `criterion_evaluation`** — wire `criteria_definition` → N parallel `criterion_evaluation` invocations via `Send`, mirroring `equity_screening`'s ticker fan-out. Requires the Criteria Evaluation Agent listed under Phase 2 (currently the orchestrator that would fan criteria doesn't exist).
- [ ] **Resumable-thread subagents** — replace the stateless `task` tool flow with per-subagent persistent LangGraph threads keyed by `(parent_run_id, subagent_name)`. Parent re-issuing a task on the same key resumes the subagent's checkpoint instead of cold-booting. Defer until `SubagentRefinementMiddleware`'s scratch-cache protocol is proven insufficient in production traces.
- [ ] **Subagent-architecture rebalancing** — narrow ReAct subagents are expensive per spawn for light tool calls. Options: (a) coarser groupings, (b) `LLMToolSelectorMiddleware` flattening with the orchestrator owning all tools, (c) better orchestrator prompt that batches calls into fewer parallel `task` invocations. Defer until W7/W8 land in production and we can measure real per-spawn cost.
- [ ] **Langfuse migration of observability hooks** — when the user migrates from LangSmith to Langfuse, revisit the "no custom LangSmith tags" decision — emit Langfuse-native trace metadata for `lesson_count`, `cache_hit_rate`, `subagent_refinement_call_id` so dashboards can filter on adaptation events.
- [ ] **Tool-knowledge cross-thread sharing audit** — lessons currently namespace at `("tool_lessons", tool_name)` (system-wide). For deployments where an agent's tool failures depend on user-specific config (e.g. user-supplied API keys), evaluate switching to `("tool_lessons", user_id, tool_name)` to prevent cross-user lesson contamination.
- [ ] **Model retry — verify mid-stream retry actually fires in production** — the analysed trace pre-dates `ModelRetryMiddleware` so we never confirmed the retry path is hit on real OpenRouter `:free` errors. Capture a fresh trace (or synthetic injection test) once the W2 paid-model defaults are in.
- [ ] **Evaluate replacing `ModelFallbackMiddleware` + `with_fallback_models(*fallbacks)` with chat-model-level `with_fallbacks`** — the new `ModelConfiguration.get_chat_model_for_role(config, role)` returns a pre-composed `RunnableRetry(RunnableWithFallbacks(primary, fallbacks))` ready for direct `ainvoke`. ALL 35 builder callsites across the repo currently do `primary, *fallbacks = get_llm_for_role(role); MuffinAgentBuilder(primary).with_fallback_models(*fallbacks)` purely because the builder needs the split shape for `ModelFallbackMiddleware`. If we instead let `MuffinAgentBuilder` accept a pre-composed `Runnable` (or just call `get_chat_model_for_role` inside the builder), we could collapse the boilerplate to one line and drop `ModelFallbackMiddleware` entirely. **Trade-offs**: (a) need to drop / reduce the universal `ModelRetryMiddleware` to avoid stacking 3×3 = 9 attempts per failure on top of the chat model's own retry; (b) fallback semantics shift from "retry budget per model + middleware-level switch" to "retry whole fallback chain + invocation-level switch via `with_fallbacks`"; (c) trace data changes shape — `ModelFallbackMiddleware` events disappear, fallback transitions live inside `RunnableWithFallbacks.ainvoke` spans; (d) tool binding through `RunnableWithFallbacks.__getattr__` distributes `bind_tools` across primary + fallbacks at build time (verified in `langchain_core/runnables/fallbacks.py:593-646`) so the ReAct loop should behave the same. **Requires** integration tests against real providers (OpenAI / Anthropic / OpenRouter) confirming tool-binding + retry behaviour are preserved before committing to the migration.

### Multi-provider chains — Ollama Cloud primary + OpenRouter fallback (2026-07)

- [x] **Cross-provider `llm_chain`** — added an ordered `list[LLMProfile]` (`{provider, model, api_key?, base_url?, requests_per_second?}`) to `ModelConfiguration` (env `LLM_CHAIN` = JSON array, parsed by a before-validator that bypasses `BaseConfiguration`'s comma-split for `[`-prefixed values). Index 0 = primary, rest = fallbacks. When set it's authoritative for every role (supersedes `llm_provider`/`model` + the per-role chains) and supplies the single-model `get_llm()` default. **Per-profile rate limits** (a relaxed Ollama primary + a 0.3 rps OpenRouter-free fallback coexist — each chat model gets its own `_shared_rate_limiter`). Profiles omit `api_key` and resolve the per-provider top-level key, so secrets stay out of the JSON blob. Provider dispatch centralised in `_build_chat_model`; `get_llm_for_role` warn-skips an unbuildable fallback (BYO single-key callers) but re-raises on an unbuildable primary.
- [x] **Ollama provider** — `ChatOllama` (langchain-ollama) over the **native `/api/chat`** API (Cloud key as a bearer header via `client_kwargs`; `ollama_base_url` default `https://ollama.com`). Chose native over the OpenAI-compat `/v1` path because `/v1` tool calling is unreliable (models emit tool-call JSON as plain text) — a dealbreaker for this ReAct/MCP stack. `ChatOllama` has no `max_retries`, so the ollama branch omits it and leans on the universal `ModelRetryMiddleware`.
- [x] **System-only 500 fix — task → first human message.** Ollama Cloud (`minimax-m3:cloud`) returns **HTTP 500 on a request with only a system message and no user turn** (verified by controlled reproduction; OpenAI/Anthropic/OpenRouter tolerate it, so the OpenRouter fallback was silently masking it — 16 Ollama errors → 4 OpenRouter successes in the analysed trace). All 26 compiled-agent-as-node stages baked the task into the runtime system prompt and invoked with an empty `messages` channel. **Renamed `with_runtime_system_prompt_template` → `with_input_prompt_template`** and reworked the middleware (`_RuntimePromptMiddleware` → `_InputPromptMiddleware`): `abefore_agent` seeds the rendered input template as the **first `HumanMessage`** (once; skipped when a human message already exists, e.g. a subagent invoked via the `task` tool — which passes the task as a `HumanMessage`, `subagents.py:665`), and the capability partials still compose onto the system message. Added a universal `_EnsureUserMessageMiddleware` guard (every agent) that injects a minimal `HumanMessage` if any model call would still be system-only. Direct LLM callers (Judge/Trader/PM/persona verdicts) already pass `[SystemMessage, HumanMessage]`, so they were unaffected. Validated: 1480 unit + 217 integration pass, plus a live run of the real `ticker_classification` agent making 6 consecutive Ollama calls with zero 500s.
- [ ] **Council rate-limit tuning for Ollama Cloud** — the 13-persona council fans out ~50 parallel calls; the primary Ollama profile currently ships with no `requests_per_second` (relaxed). If Ollama Cloud's own free-tier limits bite in production, set `requests_per_second` on the ollama profile (no code change).
- [ ] **Per-role model differentiation under `llm_chain`** — the chain applies ONE model to every role (orchestrator/collector/reasoner all use the primary). The prior per-role chains (cheap collector, stronger reasoner) are single-provider only. If per-role × per-provider differentiation is needed, extend `llm_chain` to a per-role mapping (or a `{role: chain}` shape).
- [ ] **muffin-ui structured chain editor** — the app exposes Ollama as a provider + key and a "Server default" option, but a per-user custom `llm_chain` (multi-row provider/model/rps editor) is deferred; the server `LLM_CHAIN` is the primary path for the fallback chain today.

### Other improvements
- [ ] Web search returns "Tool result too large, the result of this tool call" and got looped.
- [x] Rework tool results cache. Use ToolRuntime.store (share InMemoryStore across all agents). Add tool: `write_cached_tool_output_to_backend` that will write tool output to backend from store for further manipulations. Within middleware check if tool output is already in cache (store) and if it's return it instead of reading it from sandbox. Update prompts to teach them about new tool and workflow to manipulate with data: check cached tool output using `discover_cached_tool_outputs` -> use tool to get output schema -> use `write_cached_tool_output_to_backend` to save output in backend -> execute custom code to manipulate with output.
- [x] Explore `langchain.agents.middleware.context_editing.ContextEditingMiddleware` and `langchain.agents.middleware.summarization.SummarizationMiddleware` — both wired via opt-in `MuffinAgentBuilder.with_context_editing(...)` (default trigger 40K tokens, keep 4 most-recent tool messages) and `with_summarization(...)` (default trigger 80K tokens, keep 20 most-recent messages). Together they form a layered context-bloat defence; existing `ToolResultCacheMiddleware` + deepagents `FilesystemMiddleware` size-eviction already handle the per-tool-output side.
    - [ ] Extract from news important in the current context information only (e.g. extract sentiment, evaluate how article may affect ticket short/long-term, etc)
- [ ] Explore `langchain.agents.middleware.LLMToolSelectorMiddleware` as an alternative to many data collection subagents.
- [x] Adopted `langchain.agents.middleware.ModelRetryMiddleware` as the outermost universal middleware in `MuffinAgentBuilder._assemble_middleware`. Sits above the SDK's `max_retries` (which only covers connect-time failures before a 200 OK) so it catches errors raised from inside the SSE stream — e.g. the OpenRouter free-tier `Provider returned error` injected mid-generation. Wired with `max_retries=3`, `on_failure="error"`, exponential backoff with jitter, and a `retry_on=` callable that filters out permanent errors (`AuthenticationError`, `PermissionDeniedError`, `BadRequestError`) before matching transient ones (`APIConnectionError`, `RateLimitError`, `InternalServerError`, bare `APIError`). Knobs hardcoded; override by appending a custom `ModelRetryMiddleware` via `with_middleware(...)`. Note: wrapping the chat model directly with `Runnable.with_retry()` was rejected because `RunnableRetry` does not expose `bind_tools`, which `langchain.agents.create_agent` calls on the model.
- [ ] Utilize jinja capabilities to enrich prompt tempalates with necessary data. I think we should at least include current date.
- [ ] Design work of financial depeartment from investing/trading firm with all the specific workflows they use (heavy webcrawl and reasoning task) and created tailored agents for this.
- [ ] Add citations for the data used when analyzing it (where it comes from, which provider, which command, what period of time covered, fillings, etc)
- [x] Save information about past tool call failures in some memory, so later agent can learn from them and avoid doing faulty calls (e.g. if some provider is not setup or some call requires premium subscription) — implemented as `ToolKnowledgeMiddleware` (replaces older `ToolErrorHandlerMiddleware`). Two facets: (a) duplicate-block on identical (tool, args) pairs that previously failed permanently; (b) lesson recorder that writes per-`(tool, error_class)` lessons into the shared `BaseStore` namespace `("tool_lessons", tool_name)` and injects a `## Lessons learned …` block into the system prompt before every model call. With a configured summariser (`with_tool_knowledge(summariser)`) lessons are LLM-distilled one-liners; without one, deterministic fallback strings still accumulate. Same `(tool, error_class)` only summarises once per session.
- [ ] Integrate agent development with langfuse to analyze what changes has to be made based on observations.
- [ ] Add capability to pass pre-defined conditions
- [ ] Think about adding HITL to handle: data fetchnig failure, adjusting instructions, validating criteria, etc
- [ ] Agent self-improvement
- [ ] Add an agent to analyze stock price gainers and reason why they have grown to incorporate this knowledge later
- [x] For structure outputs explore response_format for agents — implemented in Market Regime Agent via `AutoStrategy(schema=MarketRegimeOutput)`
- [x] For cross-run memory — implemented via `MuffinAgentBuilder` fluent API (`.with_sandbox()`, `.with_short_term_memory()`, `.with_persistent_memory()`, `.with_skills(...)`) that wires `MemoryMiddleware` with `memory=["/memories/AGENTS.md"]` and a composite backend exposing `/memories/` (`StoreBackend`, per-user namespace `("memories", user_id)`), `/scratch/` (`StateBackend`, thread-scoped), and read-only `/skills/` (`FilesystemBackend`). CLI passes `--user` → `configurable={"user_id": ...}`; `user_id` is required (raises `ValueError` if missing).
- [ ] Memory follow-ups:
    - [ ] Per-ticker memory route (`/ticker-memories/` namespaced `("memories", user_id, ticker)`) — add if single AGENTS.md per user balloons or per-stock recall becomes a real need.
    - [ ] Postgres-backed `BaseStore` for self-hosted production (CLI ships `InMemoryStore`; LangGraph Platform injects managed Postgres automatically).
    - [ ] Consolidate tool-result-cache under the composite backend (today it uses `BaseStore` directly at `("cache", tool_name)`).
    - [ ] External skill roots via `make_agent_backend(skills_root=...)` — the parameter exists but no agent uses a non-default root yet.
    - [ ] Seed `/memories/AGENTS.md` with defaults (style, depth, preferred valuation methods) collected in a one-time onboarding flow.
- [ ] Explore additional Valuation methodologies:
    - [ ] Precedent Transactions — zero coverage. No M&A deal data source exists in OpenBB's MCP tools, so there's no subagent to collect transaction multiples, control premiums, or deal environment context. We'd need an external M&A data provider first. OpenBB doesn't expose M&A deal databases (that's typically Bloomberg/Capital IQ/Refinitiv territory). Without deal data, there's nothing to compute on.
    - [ ] SOTP (Sum-of-the-Parts) — schema field exists (sum_of_parts: dict | None) but is explicitly None / v1 placeholder. Would need segment-level revenue/EBITDA extraction (from 10-K MD&A), per-segment peer multiples, and a new aggregation tool. Needs segment-level financials. The regulatory-filings subagent can fetch 10-K filings, but parsing segment tables from SEC filings is a non-trivial extraction problem.
    - [ ] Residual Income — also mentioned in docs/investment-process.md but not implemented.
- [ ] Check possibility of refactoring tool result cache to get output schema of pythin tools using `function.get_output_jsonschema` (method) of decorator. Check if tool result cache. Check if we can get rid off `cached_invoke` and isntead call middleware using BaseTool or function based tool. Check how it's implemented in langchain itself.


### Unbiasing agents
- [ ] When defining data needs for criterion - agent shouldn't know about subagents available, to make sure that data needs are unbaiased
- [ ] When evaluation criterion against data agent shouldn't know about ticker or any other information except data and criterion
- [ ] When reflecting on criterion evaluation - agent shouldn't know about ticker or any other information except data and criterion. It should look only on data provided and criterion evaluation results.
- [ ] When synthesizing results from evaluated criteria - agent shouldn't know about ticker or any other information except criteria evaluation results.


### CI/CD and testing
- [x] Add full e2e integration test mocking LLM calls — reusable harness in `tests/integration/_harness/` (`ScriptedChatModel`/`patch_llm`, `patch_mcp`, `patch_sandbox`, `patch_embeddings`) + per-tool fixtures + hybrid live-capture; two worked examples (`test_equity_price_collector.py` passing, `test_persona_peter_lynch.py`) + an enforcing meta-test over `langgraph.json`. Guide: [`docs/integration-testing.md`](docs/integration-testing.md).
- [x] **Fixed the systemic compiled-subagent composition bug surfaced by the integration suite** (broke council + all trading_decision graphs + persona CLI end-to-end; masked because graph tests stub the subagents). Two layers fixed: (1) every `add_node(..., input_schema=agent.input_schema)` now passes an explicit field-based TypedDict (`PersonaInput` in `personas_council/schemas.py`, `AnalystInput` in `trading_decision/state.py`) — `create_agent`'s `.input_schema` is a property-less `RootModel` that maps `{}` and raises at coercion; (2) persona/specialist subagent-produced fields flipped from `OmitFromSchema(input=True, output=True)` → `output=False` so the auto-unpacked `structured_response` values propagate to the downstream compute node. Locked by `test_persona_peter_lynch.py` (un-xfailed), `test_council_graph_e2e.py` (all 13 personas → judge), and `test_trading_analysts_e2e.py`. Composition rules documented in CLAUDE.md.
- [x] Registered the full trading-decision pipeline as the deployable `"trading_decision"` graph in `langgraph.json` via a config-only `graph.py:make_graph` factory (mirrors `council`'s `make_graph`; the Platform factory protocol rejects a `BaseStore` param). Shipped E2E coverage `tests/integration/test_trading_decision.py` (real compiled graph through `make_graph`, only LLM/MCP/sandbox mocked, schema-routed model for the parallel analysts) and moved it into `COVERED_GRAPHS`.
- [ ] Backfill E2E integration coverage for the deployable graphs still in `PENDING_INTEGRATION_COVERAGE` (`stock_evaluation`, `research`) — `council`, `trading_decision`, and `criteria_analysis` are now COVERED. Move each into `COVERED_GRAPHS` as its `tests/integration/test_<id>.py` lands.
- [ ] Add github actions to run integration tests with agents before merging pull requests (`pytest -m integration`).

## Phase 4

### Agent Evaluations
- [ ] Setup evaluation datasets best on the past stock performances and point of time evaluation
- [ ] Definition evaluation metrics
- [ ] Setup LLM-as-a-judge scoring (Explore how to callibrate it)
- [ ] Setup evaluations with Langfuse
- [ ] Optimize prompts based on evals using langfuse

### Deployment
- [x] Self-hosted infrastructure setup using [oracle-cloud-docker-swarm-setup](https://github.com/gururafiki/oracle-cloud-docker-swarm-setup) — Terraform + Ansible deploy the full 13-service stack to a single ARM Always-Free OCI VM (single-node Docker Swarm), behind Traefik (Cloudflare DNS-01 wildcard TLS) with only the chat UI + LangGraph API exposed. Templates: `docker-stack/templates/muffin/`, playbook `ansible/muffin_stack.yml`. App-level auth added via [`auth.py`](../auth.py) (`langgraph.json` `auth` key). Walkthrough in [docs/deployment.md](docs/deployment.md#oracle-cloud-always-free-docker-swarm).
    - [x] Terraform + Ansible to spin up an instance with Docker Swarm (ARM A1.Flex, 4 OCPU / 24 GB)
    - [x] Deploy Postgres (langgraph-postgres) + Agent server + client web app (agent-chat-ui) onto the Swarm
    - [x] Cloudflare as IaC via the Terraform `cloudflare` provider (`terraform/cloudflare.tf`): proxied DNS (`muffin`/`muffin-api`.rafiki.guru) + Zero-Trust Access app/policy/service-token; one CF API token drives Terraform + Traefik DNS-01
    - [x] GitHub Actions to **build & push** the 3 ARM64 images to GHCR on merge (`muffin-agent/.github/workflows/build-images.yml`, native `ubuntu-24.04-arm` runner)
    - [ ] GitHub Actions to **auto-deploy** to the swarm on merge — add a deploy job that SSHes to the VM and rolls services (`SSH_HOST` + `SSH_PRIVATE_KEY` repo secrets)
    - [ ] Spin up independent test and prod swarms (Terraform workspaces + `<workspace>.tfvars`)
    - [ ] Deploy [langfuse](https://langfuse.com/self-hosting) onto the swarm for tracing
- [ ] **Migrate the deployed server to [Aegra](https://github.com/aegra/aegra)** (postponed 2026-06-19; ~985★, Apache-2.0, Agent-Protocol compatible). Removes the LangGraph standalone server's `LANGSMITH_API_KEY` dependency + the **1M-node-execution ceiling** (a real ceiling for muffin's node-heavy multi-agent runs) and brings auth (JWT/OAuth) OOTB. Plan: (a) validate locally with `aegra dev` against muffin's graphs (council / trading_decision / research / investment) + custom middleware / MCP / HITL; (b) swap the `langgraph-api` service + image build to Aegra (`aegra.json`, `aegra serve`) and replace [`auth.py`](../auth.py) with Aegra auth. ~90% of the deploy (Traefik/Swarm/PG/Redis/Cloudflare/Ansible) is reused.
- [ ] **Deployment hardening / known limitations** (current single-node setup):
    - [ ] Single node = no HA; add scheduled `pg_dump` backups of the `langgraph-data` volume (or move to managed/Supabase Postgres)
    - [ ] Encrypt app API keys at rest — they're plaintext in the Swarm service spec today (LangGraph reads plain env); add an entrypoint that sources Docker secret files into env, or get it for free via the Aegra migration
    - [ ] Verify ARM64 manifests for all third-party images; the OpenSandbox runtime images (`execd`/`egress`) are the most likely arm64 gap — degrade `execute_python` gracefully or pin/replace
    - [ ] Swarm drops `depends_on` startup ordering — services converge via restart; consider a lightweight wait/init if cold-start churn is noisy
    - [ ] Lock the OCI security list to Cloudflare IP ranges so the origin isn't directly reachable
    - [ ] Validate SSE (`/runs/stream`) isn't buffered/timed-out behind the Cloudflare proxy
    - [ ] Move off `nvidia/...:free` model routes for the always-on prod deploy (CLAUDE.md flags `:free` as rate-limited / mid-stream-buggy)
- [ ] Monitoring & alerting
- [ ] Scale testing

### Interface development
- [ ] Expose agents as API (LangGraph Server default API or wrap graph invocation with FastAPI)
- [ ] Expose agents as MCP servers (LangGraph Server default API or wrap graph invocation with FastMCP)
- [ ] Developing client app(s):
    - [ ] React Native cross-platform app for iOS, Android and Web.
        - Check (Vercel AI SDK)[https://ai-sdk.dev/docs/getting-started/expo]
        - Check (Gifted Chat)[https://github.com/FaridSafi/react-native-gifted-chat]
        - If costly we can start with web-only app based on React + CopilotKit.
    - [ ] Messengers

#### Sandbox
- [ ] Move Sandbox to separeate package: `langchain_opensandbox`
- [x] Generalize `write_tool_output_to_backend` — renamed to `write_cached_tool_output_to_backend`, added generic store CRUD tools (`tools/store.py`), sandbox↔store bridge tools (`sandbox/tools.py`), `StoreConfiguration` namespace access control (`utils/store_config.py`), `ToolResultCacheConfiguration` for configurable schema scanning, restructured prompt partials into three layers (`middlewares/tool_result_cache.jinja`, `sandbox.jinja`, `tools/store.jinja`)
- [ ] Explore `context_schema` to store sandbox id/thread id: https://docs.langchain.com/oss/python/langchain/tools#context
- [ ] Keep in memory/readme already written scripts.
- [x] Auto-recreate dead sandboxes — `SandboxFactory` discovers by `thread_id` metadata and creates if not found
- [x] ~~External DB for thread_id→sandbox mapping~~ — solved by OpenSandbox metadata API (`SandboxFilter(metadata={"thread_id": ...})`)
- [ ] Share scripts between agent calls.
- [ ] Once authentication is enabled - store scripts per user in persistent storage and pre-populated sandboxes with them.
- [ ] Think about having separate Coding agent instead of writing scripts within each agent.
- [ ] Migrate `SandboxFactory` from sync to async `opensandbox` client so `langgraph dev` no longer needs `--allow-blocking` (sync `socket.connect` is flagged under ASGI).
- [ ] Make `opensandbox-server` advertise a dual-addressable sandbox endpoint (loopback for host callers + `host.docker.internal` for container callers) so the dev workflow no longer needs a separate `config.dev.toml` + [docker-compose.dev.yml](docker-compose.dev.yml) overlay. When done, delete [extras/opensandbox/config.dev.toml](extras/opensandbox/config.dev.toml) and [docker-compose.dev.yml](docker-compose.dev.yml), and drop the `-f docker-compose.dev.yml` flag from [.vscode/tasks.json](.vscode/tasks.json).

### DX
- [ ] Add Claude Code skills for Spec driven development
- [x] Check Claude Code development via mobile app


## Persona Council (ai-hedge-fund port)

Ported from [ai-hedge-fund](https://github.com/virattt/ai-hedge-fund).  Guide: [`docs/personas.md`](docs/personas.md).

### Council
- [x] Phase 1 foundations — `AnalystSignal`, `PersonaDataBundle`, shared scoring helpers (`agents/personas_council/tools/scoring_helpers.py`), technical-indicator + sentiment tools, registry scaffolding
- [x] Phase 2 — 13 personas (Buffett, Graham, Wood, Munger, Ackman, Burry, Pabrai, Taleb, Lynch, Fisher, Jhunjhunwala, Druckenmiller, Damodaran) + council graph (`build_council_graph`) + LLM-mediated judge synthesis + `muffin persona` / `muffin council` CLI + LangGraph registration as `"council"`
- [x] Phase 3 — Technical-analysis + sentiment-analysis specialists (deterministic, no LLM) + `SPECIALIST_REGISTRY` + `muffin technicals` / `muffin sentiment` CLI
- [x] Phase 3b — Ported the 4 remaining ai-hedge-fund specialists: `fundamentals`, `growth`, `valuation`, `news_sentiment` (persona-style ReAct extraction → deterministic scoring) + `tools/{fundamentals,growth,valuation_signal}.py` + council `include_specialists` wiring + CLI (`growth` / `valuation` / `news-sentiment` / `fundamentals-signal`)
- [x] Phase 3c — Restored persona valuation/DCF parity with ai-hedge-fund: history-derived growth + exit-multiple terminals (Buffett / Ackman / Rakesh), relative-median P/E + base-FCFF terminal (Damodaran), full Buffett owner earnings (median maintenance capex + ΔWC); discrete fixes (Munger ≥5% buyback gate, Pabrai FCF trend, Taleb population stats, Buffett confidence-anchor prompt, `metrics_history` oldest→newest)
- [x] Phase 3d — Consolidated the whole port into `agents/personas_council/` (personas / specialists / tools / portfolio); council registered as the async **factory** `council_graph.py:build_council_graph` (config-driven `include_specialists`); collectors that derive ratios get `execute_python`; **removed** the graceful LLM-failure `hold` fallbacks (prefer fail-loud + retries over a hidden empty result)

### Paper-trading + backtester (removed → deferred to a generic module)
- **Removed** the council-specific paper-trading layer (`personas_council/portfolio/`: Portfolio book + executor, `risk_management_node`, `ticker_decision_node`, `portfolio_reconciler_node`, the `build_portfolio_decision_graph` graph) + the `muffin trade` CLI, **and** the council-specific `BacktestEngine`. Both were ai-hedge-fund-derived and council-coupled; upstream the Portfolio book + `TradeExecutor` were backtester infra and the decision logic (vol×corr sizing, order-legality reconciliation) is generic, not council-specific. The orphaned `portfolio_reconciler.jinja` / `ticker_decision.jinja` prompts and the unused `utils/outcomes.py` re-export shim were since removed too (2026-06-18) — the `OutcomesFetcher` Protocol + `fetch_decision_outcome` live solely in `trading_decision/tools.py`; the future generic module below can import from there directly.
- [ ] Build a **generic, pipeline-agnostic portfolio-management + paper-trading + backtester** module usable by council / `trading_decision` / `investment`: Pydantic Portfolio book + executor (partial-fill, margin), vol×corr position sizing, order-legality reconciliation, and a walk-forward backtester (`full`/`signals` modes, Sharpe/Sortino/max-drawdown/SPY metrics). Harden as-of-date enforcement on OpenBB MCP tools (server-side `curr_date` cap) for reproducible historical runs. Reference impls (the removed council paper-trading layer + ai-hedge-fund `BacktestEngine`) are in git history.

### Open items
- [x] News sentiment LLM-per-article specialist — ported (Phase 3b, `news_sentiment`)
- [x] Growth-analysis specialist — ported (Phase 3b, `growth`)
- [ ] Port Nassim Taleb's `analyze_black_swan_sentinel` (negative-news ratio + volume-spike crisis signal) — deferred; needs company-news collection added to the Taleb data step (volume/price-dislocation half already implicit in the verdict via tail-risk + vol-regime evidence)
- [ ] Damodaran terminal value uses upstream's base-FCFF anchor (understates IV) for parity; revisit whether to switch to the textbook `terminal_basis="final_cf"` as the default
- [ ] Wire personas into `investment_analysis.py` as an opt-in stage (currently they're independent + pluggable via `PERSONA_BUILDERS`)
- [ ] **Validate collector code-execution + tool-result-cache need.** Run real councils and evaluate the quality/correctness of collector arithmetic (gross_margin / BVPS derivations, series ordering). Decide whether the `execute_python` tool now on the persona + fundamentals/growth/valuation collectors is justified (vs. moving those derivations into the deterministic compute nodes), and whether to add the tool-result-cache workflow via `.with_sandbox()` (cache tools + `tool_result_cache.jinja` partial + offload backend) so collectors can discover sibling-cached data and process large/offloaded outputs in code — matching the `equity_price` collector pattern.
- [ ] Council debate mode using `multi_agent` conference framework (sequential deliberation alternative to parallel vote)
- [ ] Consider using [skills](https://docs.langchain.com/oss/python/deepagents/skills) per each persona with python scripts to compute necessary scores instead of having scoring functions as utils manually invoked from nodes.
## Multi-repo restructure (Done — 2026-06)

Split into focused repos under the umbrella [`muffin`](https://github.com/gururafiki/muffin) (git submodules): `muffin-agent` (this repo — agent + image CI), [`muffin-deployment`](https://github.com/gururafiki/muffin-deployment) (Terraform + Ansible + Swarm stack + service configs + deploy CI), and per-service image repos `openbb-mcp-docker`, `agent-chat-ui-docker`, `nuq-postgres-docker` (each builds its own arm64 GHCR image). The local `docker-compose`, `extras/*` Dockerfiles, and service configs moved out of this repo into `muffin-deployment` / the `*-docker` repos. Deployment is a single `terraform apply` (runs Ansible via the `ansible/ansible` provider + `cloud.terraform` dynamic inventory — no `generate_inventory.sh`). Live: https://muffin.rafiki.guru.
