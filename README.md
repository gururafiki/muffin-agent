# 🧁 Muffin Agent

**A hierarchical multi-agent system for comprehensive stock analysis using LangGraph**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2.6+-green.svg)](https://github.com/langchain-ai/langgraph)
[![License: GNU GPL v3](https://img.shields.io/badge/License-GNU_GPL_v3-yellow.svg)](LICENSE)


## 🎯 Overview

Muffin Agent is a production-ready, multi-agent stock analysis system that functions as a complete investment research department. Built with LangGraph, it orchestrates specialized agents that analyze stocks from multiple perspectives—technical, fundamental, news sentiment, strategic positioning, and competitive landscape—to produce comprehensive investment theses with price targets.

### Key Features (In development)

- **🤖 Multi-Agent Architecture**: Specialist agents working in parallel with cross-validation
- **📊 Technical Analysis**: RSI, MACD, Bollinger Bands, Moving Averages, Volume Analysis
- **💰 Fundamental Analysis**: Financial metrics, growth rates, profitability, balance sheet health
- **📰 News & Sentiment**: Market sentiment, catalysts, social signals
- **🎯 Multi-Timeframe Targets**: Price targets for 1m, 3m, 6m, 1y, 3y horizons
- **✅ Structured Outputs**: Type-safe Pydantic models throughout
- **🔄 Graceful Degradation**: Continues with partial agent failures
- **🔌 Multi-LLM Support**: OpenAI, Anthropic, OpenRouter
- **🆓 Free Data Sources**: OpenBB (free tier), yfinance, SEC Edgar
- **🔧 MCP Integration**: Data collection agents use OpenBB MCP tools with configurable tool subsets per agent

### Data Collection Agents

ReAct agents that retrieve financial data via OpenBB MCP. Each agent has a filtered subset of tools and can be extended with custom `@tool` functions.

| Agent | Tools | Description |
|-------|-------|-------------|
| `equity_fundamentals` | 25 | Financial statements, ratios, metrics, EPS, dividends, revenue segments, management, ESG, transcripts, filings |
| `equity_price` | 5+1 | Current quotes, historical OHLCV, NBBO spreads, price performance, market cap history. Also includes `execute_python` for in-sandbox computations (DCF, technical indicators) |
| `equity_estimates` | 8 | Analyst consensus estimates, price targets, forward EPS/EBITDA/PE/sales, analyst rating breakdowns |
| `equity_ownership` | 9 | Major holders, institutional ownership, insider trading, share statistics, 13F filings, government trades, short interest/volume/FTDs |
| `news` | 2 | Company news with sentiment signals, global/macro news headlines |
| `options` | 2 | Options chains with Greeks (delta, gamma, theta, vega, IV), implied volatility surface |
| `economy_macro` | 40 | GDP, CPI, unemployment, interest rates, FOMC documents, FRED series, surveys (UMich, SLOOS, payrolls, manufacturing), shipping volumes |
| `fixed_income` | 24 | Interest rates (SOFR, EFFR, ECB, SONIA), yield curves, Treasury rates/prices, TIPS, corporate bonds, spreads, mortgage indices |
| `etf_index` | 19 | ETF info/sectors/holdings/returns, index levels, S&P 500 multiples, reverse ETF lookup by stock ticker |
| `discovery_screening` | 23 | Equity screener, gainers/losers/active, earnings/IPO/dividend calendars, peer comparisons, sector group valuations, company profiles, dark pool |
| `currency_commodities` | 9 | FX pair history and reference rates, commodity spot prices (WTI, Brent, gold), EIA energy outlook, crypto price history |
| `regulatory_filings` | 14 | SEC filings, CIK lookups, CFTC Commitment of Traders, US congressional bills |
| `fama_french` | 6 | Fama-French 3/5-factor model returns, US/regional/country portfolio returns, international index returns, size/value breakpoints |
| `web_search` | 6 Firecrawl MCP + SearxNG + 1 custom | Web search (SearxNG via `load_tools`), Firecrawl scrape/crawl/map/batch-scrape/extract, document-to-Markdown (MarkItDown) |

### Data Validation Agent

A pure reasoning agent (no tools) that validates collected financial data against a given criterion. Used as a subagent by both the Stock Evaluation Agent and the Criterion Evaluation Agent (Step 3 in each workflow).

It scores data across four dimensions (each 0.0–1.0):

| Dimension | What it checks |
|-----------|---------------|
| Sufficiency | Are key data points present and usable for the criterion? |
| Relevance | Does the data directly address the criterion being evaluated? |
| Temporal Validity | Does all data respect the analysis date cutoff? |
| Consistency | Do units, periods, and currencies match across sources? |

**Output**: Structured report with per-dimension scores, weighted overall confidence (0.0–1.0), overall relevance, identified gaps/issues, and a recommendation: `proceed`, `collect_more_data` (with specific gaps to fill), or `insufficient_data`.

### Stock Evaluation Agent

A deep agent (powered by `deepagents`) that orchestrates all 13 data collection subagents plus a data validation subagent to produce scored stock assessments. Subagents are created via the shared `build_analysis_subagents()` helper in `agents/subagents.py`. It follows a 5-step workflow:

1. **Plan** — Determine what data is needed based on ticker and query
2. **Collect** — Delegate to data collection subagents via `task()` tool
3. **Validate** — Check data sufficiency, relevance, temporal correctness, completeness
4. **Analyze** — Produce a 0.0–1.0 score with reasoning backed by specific data points
5. **Reflect** — Verify score-data consistency, logical coherence, and confidence

**Sandbox isolation**: Each conversation gets its own OpenSandbox container. Sandboxes are discovered lazily by `thread_id` metadata — `get_backend` and `execute_python` find or create a container for the current conversation via the OpenSandbox API. If a container dies mid-conversation, a new one is created transparently. Parallel conversations never share execution state.

### Criterion Evaluation Agent

A deep agent that evaluates a **single investment criterion** (e.g., "Does the company have strong profitability?", "Is the balance sheet healthy?") by collecting targeted data, validating it, and producing a scored assessment. Uses the same shared subagents as the Stock Evaluation Agent. It follows a 5-step workflow:

1. **Analyze Criterion** — Parse the criterion, determine data needs, select 2-4 relevant subagents using the built-in selection guide
2. **Collect Data** — Delegate to selected data collection subagents with specific, targeted requests
3. **Validate Data** — Delegate to the data-validation subagent; iterate up to 2 times if gaps are found
4. **Evaluate** — Decompose the criterion into 2-4 dynamic sub-criteria, score each 0.0–1.0 using Chain-of-Thought with formula-first calculations, then combine into a weighted overall score
5. **Reflect** — Check for score-evidence consistency, confirmation bias, anchoring bias, and missing counterarguments

**Output**: Structured `CRITERION_EVALUATION_START/END` delimited output with score, confidence (numeric 0.0–1.0), signal, sub-criteria breakdown, evidence summary, reasoning, counterargument, and limitations. Designed to be consumed by the parent Criteria Evaluation Agent (planned).

### Criteria Definition Agent

A standalone deep agent that classifies a ticker by sector, market type (developed/emerging), and stock type (value/growth), then loads matching valuation skills via progressive prompt disclosure to produce sector-specific evaluation criteria with target ranges and methodology guidance.

Uses 5 subagents (etf-index, equity-fundamentals, discovery-screening, economy-macro, data-validation) and 55 valuation skills organised under `skills/valuation/`. Skills are pre-filtered by `SkillFilterMiddleware` based on a flat classification provided as input state — the agent only sees the 4-6 skills relevant to its classification.

**Workflow**: Parse Context → Collect Data (4 subagents in parallel) → Validate → Load Skills & Extract Criteria → Reflect.

**Output**: `CriteriaDefinitionOutput` with ticker, sector, market type, stock type, 5-8 valuation criteria (with target ranges, weights, and guidance), screening questions, and valuation pitfalls.

**CLI**:
```bash
muffin criteria AAPL --sector banking --market developed --stock-type value
```

### Market Regime Agent

A deep agent (Step 2 of the investment process) that classifies the current macro and liquidity regime, identifies factor tailwinds and headwinds, and provides portfolio positioning guidance. Uses a **macro-focused subset** of 6 subagents — the 5 market-wide data collection agents plus data-validation — built by a private `_build_macro_subagents()` helper rather than the full 14-agent set.

Supports three context modes via the `MarketRegimeContext` TypedDict:

| Mode | Fields | Behaviour |
|------|--------|-----------|
| Ticker | `ticker` | Calls `etf_equity_exposure` to derive sector/style; populates `ticker_impact` in output |
| Explicit context | `sector`, `industry`, `country` | Uses supplied context directly |
| Query-only | `query` | Narrows geographic or style focus via investment mandate text |

Follows a 5-step workflow: **Parse Context → Collect Macro Data → Validate → Classify Regime (4 dimensions) → Reflect**.

**Four regime dimensions** are scored and labelled:

| Dimension | Scale | Labels |
|-----------|-------|--------|
| Growth / Activity Cycle | 0 = deep contraction → 1 = strong expansion | `expanding`, `slowing`, `contracting`, `recovering` |
| Inflation / Price Regime | 0 = deflation → 1 = severe inflation | `high_rising`, `elevated_stable`, `moderate`, `low_falling`, `deflationary` |
| Monetary Policy Stance | 0 = aggressively easing → 1 = aggressively tightening | `aggressively_tightening`, `tightening`, `neutral`, `easing`, `aggressively_easing` |
| Liquidity / Risk Appetite | 0 = crisis → 1 = extreme risk-on | `risk_on`, `cautiously_risk_on`, `neutral`, `risk_off`, `crisis` |

**Structured output** is enforced via `response_format=AutoStrategy(schema=MarketRegimeOutput)` — the LLM calls a structured output tool as its final act, returning a validated `MarketRegimeOutput` Pydantic instance in `result["structured_response"]`. No regex parsing. The `market_regime_node` calls `.model_dump()` to store the result in graph state.

**Output** (`MarketRegimeOutput`) includes: `regime_label`, `as_of_date`, `confidence`, `dimensions`, `factor_assessment` (value / quality / momentum / size tilts), `yield_curve`, `macro_summary`, `key_risks`, `recommended_positioning` (beta range, gross/net exposure, sector/style tilts), optional `ticker_impact`, `data_sources`, and `limitations`.

Used in both `TickerAnalysisState` (per-ticker analysis) and `ScreeningState` (shared pre-fanout context for equity screening).

### Sector Analysis Agent

A deep agent (Step 3 of the investment process) that assesses the attractiveness of a sector/industry: competitive structure (Porter's Five Forces), cycle position, thematic tailwinds/headwinds, sector-relative valuation, regulatory/legislative backdrop, and alpha opportunity (peer return dispersion). Uses a **sector-focused subset** of 6 subagents — etf-index, discovery-screening, equity-estimates, news, regulatory-filings, and data-validation — built by a private `_build_sector_subagents()` helper. The equity-estimates subagent provides earnings revision breadth (% of peers with upward vs downward EPS revisions) as a leading cycle position indicator.

Supports the same three context modes as the market regime agent via `SectorAnalysisInputState`:

| Mode | Fields | Behaviour |
|------|--------|-----------|
| Ticker | `ticker` | Calls `etf_equity_exposure` to derive sector/industry |
| Explicit context | `sector`, `industry` | Uses supplied context directly (screening graph pre-fanout) |
| Query-only | `query` | Infers sector focus from the investment mandate text |

Follows a 5-step workflow: **Parse Context → Collect Sector Data → Validate → Score Sector (6 dimensions) → Reflect**.

**Six scored dimensions:**

| Dimension | Output |
|-----------|--------|
| Cycle Position | `early_expansion` \| `mid_expansion` \| `late_cycle` \| `contraction` \| `recovery` + direction |
| Competitive Structure | Full Porter's Five Forces (rivalry, barriers, supplier power, buyer power, substitutes) + `overall_attractiveness` |
| Thematic Drivers | Structured list of themes with `direction` (tailwind/headwind/neutral) and `time_horizon` |
| Sector Valuation | P/E and EV/EBITDA vs. S&P 500 (sandbox-computed) → `expensive` \| `fairly_valued` \| `cheap` |
| Regulatory Backdrop | `low` \| `moderate` \| `elevated` \| `high` + specific legislative/enforcement items |
| Alpha Opportunity | `high` \| `moderate` \| `low` based on peer return dispersion (`compute_peer_dispersion` tool) |

**Financial tools**: `compute_sector_relative_performance` (called 3× for 1M/3M/12M horizons), `compute_peer_dispersion` (peer return std dev). Sandbox remains for P/E premium/discount calculation.

**Structured output** is enforced via `response_format=AutoStrategy(schema=SectorViewOutput)`. Output includes: `sector`, `industry`, `cycle_position`, `competitive_assessment`, `thematic_drivers`, `sector_valuation`, `regulatory_backdrop`, `peer_tickers`, `alpha_opportunity`, `alpha_rationale`, `sector_signal` (favorable/neutral/cautious), `sector_summary`, `confidence`, `data_sources`, `limitations`.

Runs in parallel with `market_regime_node` and `company_analysis_node` (Group 1). Also used as a shared context node in `ScreeningState` before the per-ticker fan-out.

### Company Analysis Agent

A deep agent (Steps 4–5 of the investment process) covering Business/Moat/Management/ESG Triage and Financial Quality Deep Dive. Uses a **company-focused subset** of 6 subagents — equity-fundamentals, equity-ownership, regulatory-filings, news, discovery-screening, and data-validation — built by a private `_build_company_analysis_subagents()` helper.

Supports two context modes via `CompanyAnalysisInputState`:

| Mode | Fields | Behaviour |
|------|--------|-----------|
| Ticker | `ticker` + `query` | Standard per-ticker company analysis |
| Query-only | `query` | Thematic quality screen without a specific ticker |

Follows a 5-step workflow: **Parse Context → Collect Data → Validate → Assess Company (4 dimensions) → Reflect**.

**Four assessment dimensions:**

| Dimension | Output |
|-----------|--------|
| Competitive Moat | `width` (wide/narrow/none/negative), `sources` list, `trend`, `confidence` with peer ROIC premium |
| Management Quality & Capital Allocation | `track_record`, `capital_allocation_quality`, `insider_alignment`, `key_concerns` |
| ESG & Governance Triage | `esg_signal` (green/amber/red), `esg_flags` list |
| Financial Quality | `quality_signal` (high/adequate/low/distressed), `trend` based on margin evolution |

**Financial tools** (all mandatory): `compute_roic`, `compute_fcf_conversion`, `compute_net_debt_to_ebitda`, `compute_interest_coverage`, `compute_altman_z_score` (financial distress indicator: >2.99 safe, 1.81–2.99 grey zone, <1.81 distress), `compute_revenue_cagr` (3Y), margin trends. Peer ROIC premium computed via sandbox.

**Financial history**: Builds up to 10 years of parallel time-series arrays (revenue, gross profit, EBIT, EBITDA, net income, FCF, capex, total debt, cash, working capital, total assets, shareholders' equity) for downstream `forecasting_node` calibration.

Issues a **triage gate signal** (`company_signal`: pass/watch/fail) that determines whether the idea advances to forecasting and valuation, and anchors scenario probability weights in the forecasting model.

**Structured output** is enforced via `response_format=AutoStrategy(schema=CompanyAnalysisOutput)`. Output includes: `ticker`, `company_name`, `business_description`, `moat_assessment`, `management_quality`, `esg_flags`, `esg_signal`, `financial_quality`, `capital_allocation_summary`, `key_risks`, `financial_history`, `company_signal`, `quality_summary`, `confidence`, `data_sources`, `limitations`.

Runs in parallel with `market_regime_node` and `sector_analysis_node` (Group 1). Its output is the primary input for `forecasting_node` and `risk_assessment_node` (Group 2).

### Forecasting Agent

A deep agent (Step 6 of the investment process) that builds a 3-year forward **three-statement financial model** (income statement, cash flow, and balance sheet) with bull, base, and bear scenarios anchored to analyst consensus. Uses a **forecasting-focused subset** of 4 subagents — equity-estimates, equity-fundamentals, economy-macro, currency-commodities, and data-validation — built by a private `_build_forecasting_subagents()` helper.

Supports three context modes via `ForecastingInputState`:

| Mode | Fields | Behaviour |
|------|--------|-----------|
| Full pipeline | `ticker` + `query` + `company_analysis` + `market_regime` | Uses `financial_history` from company_analysis as baseline; applies macro regime context from market_regime |
| Ticker + query | `ticker` + `query` | Fetches all historical financials fresh from equity-fundamentals |
| Query-only | `query` | Thematic or sector-level forward model without a specific ticker |

Follows a 5-step workflow: **Parse Context → Collect Data → Validate → Build Scenarios → Reflect**.

**Financial tools** (all mandatory): `project_three_year_financials` (3-year income statement, FCF, and balance sheet projection per scenario — called 3× in parallel for bull/base/bear), `compute_sensitivity` (±1pp revenue/margin impact on EPS and FCF), `compute_accruals_ratio` (earnings quality), `compute_revenue_cagr` (historical calibration). Sandbox (`execute`) remains for historical array processing, consensus revision momentum, and balance sheet tie validation.

**Three scenarios with explicit probability anchors:**

| Signal | Base | Bull | Bear | Notes |
|--------|------|------|------|-------|
| `pass` | 0.60 | 0.25 | 0.15 | Default; LLM can deviate with written rationale |
| `watch` | 0.50 | 0.25 | 0.25 | Increased bear weight for watch-listed companies |
| `fail` | 0.40 | 0.25 | 0.35 | Full analysis still runs; useful for short theses |

**Structured output** is enforced via `response_format=AutoStrategy(schema=ForecastOutput)`. Output includes: `base_case`, `bull_case`, `bear_case` (each with `list[YearlyProjection]` for Y+1/+2/+3 covering revenue, EBITDA, EBIT, EPS, FCF, and balance sheet items: net_debt, total_debt, cash, working_capital, total_assets, shareholders_equity), `consensus_anchoring` (consensus EPS/revenue/EBITDA, price targets, revision_trend_3m, surprise_history), `revision_momentum`, `sensitivity_table`, `earnings_quality_flags`, `modeling_notes`, `confidence`, `data_sources`, `limitations`. EPS is null if diluted share count is unavailable. Scenario probability anchors are pre-computed from `company_signal` and injected as Jinja2 template variables.

Runs in parallel with `risk_assessment_node` (Group 2). Its output is the primary input for `valuation_node` (Group 3).

### Risk & Downside / Stress Testing Agent (Step 8)

A deep agent (Step 8 of the investment process) that quantifies idiosyncratic and systematic risk, derives stress scenarios, and produces an ex-ante stop level. Uses a **risk-focused subset** of 7 subagents — equity-price, options, fama-french, equity-ownership, fixed-income, economy-macro, and data-validation — built by a private `_build_risk_assessment_subagents()` helper.

**Five-step workflow**: (1) **Plan** — scope subagent calls and tool invocations from ticker/context; (2) **Collect** — run subagents in parallel via `SubAgentMiddleware`; (3) **Validate** — data sufficiency check (data-validation subagent); (4) **Analyse** — sandbox computes price return series, then four parallel blocks: Block A (4 parametric tools: `compute_beta`, `compute_var_cvar`, `compute_sharpe_sortino`, `compute_max_drawdown`), Block B (FF5+UMD 6-factor OLS regression via `execute_python`), Block C (IV term structure extraction), Block D (short interest crowding classification); (5) **Reflect** — self-critique and confidence calibration.

**Stress scenarios** are hybrid: 2 fixed historical analogs (GFC 2008, market −56%; COVID crash 2020, market −34%), 3 regime-derived from `market_regime.key_risks`, and 1 idiosyncratic. Each scenario reports estimated stock return and dollar impact per share.

**Four deterministic tools** in `tools/risk.py` (all Python stdlib, no scipy):
- `compute_beta` — OLS beta, annualised Jensen's alpha, R² vs. market benchmark
- `compute_var_cvar` — parametric 95% VaR and CVaR scaled to any horizon (uses `statistics.NormalDist`)
- `compute_sharpe_sortino` — annualised Sharpe and Sortino ratios; daily/weekly frequency
- `compute_max_drawdown` — peak-to-trough drawdown from a price series

**Structured output** is enforced via `response_format=AutoStrategy(schema=RiskAssessmentOutput)`. Output includes: beta, annualized_vol_pct, max_drawdown_1y_pct, var_95_1m_pct, cvar_95_1m_pct, sharpe_ratio, sortino_ratio, `factor_loadings` (FF5+UMD betas, alpha, R², regression_period), `implied_volatility` (IV 30/60/90d + 25d put/call skew + term_slope), `short_interest` (short_interest_pct, days_to_cover, short_volume_ratio, crowding_signal), `stress_scenarios` (list of 6), ex_ante_stop_level, stop_methodology, `risk_signal` (acceptable / elevated / unacceptable), risk_flags, confidence, data_sources, limitations.

Runs in parallel with `forecasting_node` (Group 2).

### Valuation & Relative Value Agent

A deep agent (Step 7 of the investment process) that computes intrinsic value via multiple methods, benchmarks against peers and history, and assigns a `valuation_signal`. Uses a **valuation-focused subset** of 5 subagents — equity-price, equity-estimates, etf-index, discovery-screening, fixed-income, and data-validation — built by a private `_build_valuation_subagents()` helper.

Follows a 5-step workflow: **Parse Context → Collect Data → Compute Valuations → Synthesize → Reflect**.

**Five compute blocks:**

| Block | Tools / Method | Purpose |
|-------|---------------|---------|
| A | `compute_wacc` | CAPM cost of equity (rf + β × ERP); WACC with debt tax shield |
| B | `compute_dcf` ×3 | Blended DCF: exit-multiple + Gordon Growth terminal value, averaged when both available; separate bull/base/bear runs |
| C | `compute_multiples_value` ×3 | EV/EBITDA, P/E, and FCF-yield implied prices |
| D | Sandbox | 5-year own-historical P/E and EV/EBITDA average vs. current |
| E | `compute_scenario_weighted_value` | Probability-weighted NAV, upside/downside %, risk-reward ratio |

**Four deterministic tools** in `tools/valuation.py`:
- `compute_wacc` — CAPM ke = rf + β × ERP; WACC = equity_weight × ke + debt_weight × kd × (1−t); validates weight sum ≈ 1.0
- `compute_dcf` — discounts 3 FCF years + blended terminal value; `methodology`: `blended` | `exit_multiple` | `gordon_growth`
- `compute_multiples_value` — enterprise value implied price via EV/EBITDA and P/E; FCF-yield intrinsic value
- `compute_scenario_weighted_value` — probability-weighted NAV, upside/downside %, risk-reward ratio

**Reflect step** validates: DCF vs. multiples within ±50%, analyst PT vs. model within ±30%, WACC in 6–15% range, scenario probabilities sum to 100%.

**Structured output** is enforced via `response_format=AutoStrategy(schema=ValuationOutput)`. Output includes: `current_price`, `dcf_value` (bull/base/bear NAV + blended base + methodology + wacc_used), `ev_ebitda_value`, `pe_value`, `fcf_yield_value`, `analyst_target_median`, `probability_weighted_nav`, `upside_base`, `upside_bull`, `downside_bear`, `risk_reward_ratio`, `relative_value` list (ev_ebitda + pe each with stock_current, peer_median, market_median, premium_discount_pct, historical_5y_avg, vs_own_history), `wacc`, `valuation_signal` (cheap/fairly_valued/expensive), `key_valuation_drivers`, `confidence`, `data_sources`, `limitations`.

Runs **sequentially** after Group 2 barrier (forecasting + risk_assessment); output consumed by `thesis_synthesis_node`.

### Middleware

Agents are composed via `MuffinAgentBuilder`, which wires universal middleware for every agent and opt-in middleware per capability. The builder is fully typed: `with_system_prompt` accepts `str | SystemMessage`, `with_permission` accepts a real `FilesystemPermission` (deep-agent only), and the tool / subagent / middleware signatures forward exactly the types expected by `create_deep_agent` and `create_agent`.

**Universal middleware** (always added):

| Middleware | Purpose | Tools |
|-----------|---------|-------|
| **`ModelRetryMiddleware`** (upstream) | Retries transient/mid-stream LLM provider errors that escape the SDK's connect-time `max_retries` (`APIConnectionError`, `RateLimitError`, `InternalServerError`, bare mid-stream `APIError`). Filtered to skip permanent errors (auth, perm, validation). 3 retries with exponential backoff + jitter, `on_failure="error"` so `ModelFallbackMiddleware` can step in. | None |
| **`ToolKnowledgeMiddleware`** | Catches tool exceptions and errored `ToolMessage`s; blocks identical-args repeats of permanent failures (`failed_tool_calls` state); records one lesson per unique `(tool, error_class)` pair into the shared store under `("tool_lessons", tool_name)`. With a configured summariser (`with_tool_knowledge(summariser)`) lessons are LLM-distilled one-liners; without one, deterministic fallback strings. `awrap_model_call` injects a `## Lessons learned …` block into the system prompt before every model call. | None |
| **`ToolResultCacheMiddleware`** | Caches successful tool results in-memory for cross-agent deduplication. On cache hit returns the cached content without re-executing. **Strict-content invariant** — tool content is byte-for-byte original; cache provenance lives in `additional_kwargs["cache"]`. Size-based offload of oversized payloads is delegated to deepagents `FilesystemMiddleware._aintercept_large_tool_result` (default 20K-token threshold). | `discover_cached_tool_outputs`, `get_tool_output_schema`, `write_cached_tool_output_to_backend` |
| **`ToolRetryMiddleware`** (upstream) | Retries transient tool errors (HTTP 5xx, gateway, network, timeout) via a custom filter that matches `ToolException` message substrings. 4xx, validation, and missing-credential errors propagate so the LLM (and `ToolKnowledgeMiddleware`) can adapt. 1 retry, `on_failure="continue"`. Sits below the cache so cache hits short-circuit and don't burn retry budget. | None |

**Opt-in middleware** (added when a specific `with_*` method is called):

| Method | Middleware added | Purpose |
|---|---|---|
| `.with_fallback_models(*models)` | `ModelFallbackMiddleware` (upstream, outermost) | Tries the primary model; on exception, walks the caller-supplied fallback chain. Each fallback model receives the same retry budget as the primary; fallback only kicks in after retries are exhausted. Pair with `ModelConfiguration.get_llm_for_role(role)` for cross-provider chains. |
| `.with_context_editing(...)` | `ContextEditingMiddleware` (upstream) | Trims older tool outputs once the message-window token count exceeds the trigger (default 40K, keep 4 most-recent). |
| `.with_summarization(...)` | `SummarizationMiddleware` (upstream) | LLM-summarises older messages when context-editing alone can't cap the window (default trigger 80K tokens, keep 20 messages). Defaults the summariser model to the agent's primary. |
| `.with_model_call_limit(...)` | `ModelCallLimitMiddleware` (upstream, outermost) | Caps total LLM calls per run/thread; `exit_behavior="end"` (default) injects a graceful `AIMessage` rather than throwing. |
| `.with_tool_call_limit(...)` and `.with_tool(..., run_limit=N, thread_limit=N)` | `ToolCallLimitMiddleware` (upstream) | Per-tool or global tool-call cap. The inline form on `with_tool(...)` declares per-tool policy at the same site as the tool itself. |
| `.with_tool_knowledge(summariser)` | (no new middleware) | Configures the always-on `ToolKnowledgeMiddleware` to use *summariser* (a small/cheap chat model) for LLM-distilled lessons instead of the deterministic fallback. |
| `.with_subagent_refinement()` | `SubagentRefinementMiddleware` (child) **or** `SubagentRefinementParentMiddleware` (parent) | Generic conversational refinement protocol. Role decided at build time: child (no subagents wired) gets the full middleware (`response_format=AutoStrategy(CollectionFindings)`, scratch-cache findings, prior-call prompt block); parent (subagents wired) gets the prompt-only parent middleware that teaches the orchestrator how to read `gaps` and re-issue with `prior_call_id=<id>`. |
| `.with_short_term_memory()` (ReAct) | `FilesystemMiddleware` | Exposes `/scratch/` file tools to ReAct agents. (Deep agents receive the composite via `backend=`.) |
| `.with_persistent_memory()` (ReAct) | `MemoryMiddleware` | Auto-loads `/memories/AGENTS.md` into the system prompt each turn. Deep agents get the same middleware implicitly via `create_deep_agent(memory=[...])`. |
| `.with_skills(..., filter_middleware=SkillFilterMiddleware[TickerClassification]())` | `SkillFilterMiddleware` | Schema-driven skill pre-filtering. Parameterised via `__class_getitem__` with an `AgentState` subclass whose extra fields become category keys. Filters `skills_metadata` and injects classification context into the system prompt. |
| `.with_middleware(m)` | Caller middleware | Appended after universal and capability middleware. |

Other middleware in the codebase (not builder-wired but available): `StoreAccessMiddleware` provides namespace-scoped CRUD access to a `BaseStore` for structured shared data — pass via `.with_middleware(StoreAccessMiddleware())` when you want the 5 store tools alongside path-addressed filesystem tools.

### Memory

Filesystem routes are composed opt-in through `MuffinAgentBuilder`:

| Prefix | Backend | Lifetime | Purpose | Enabled by |
|---|---|---|---|---|
| (no prefix) | `OpenSandboxBackend` | thread (container recycles) | `execute_python` + ephemeral workspace. Auto-offloaded tool outputs (>20k tokens) land here. | `.with_sandbox()` |
| `/scratch/` | `StateBackend` | thread (checkpointed) | Short-term notes that survive sandbox recycling. | `.with_short_term_memory()` |
| `/skills/` | `FilesystemBackend` (read-only) | static | Shipped skill files, auto-matched by built-in `SkillsMiddleware`. | `.with_skills(paths)` |
| `/memories/` | `StoreBackend`, ns `("memories", user_id)` | cross-thread, per-user | Long-term memory (AGENTS.md). Auto-loaded into system prompt by `MemoryMiddleware` (stock deepagents middleware); LLM updates via `edit_file`. | `.with_persistent_memory()` |

`user_id` is required in `RunnableConfig["configurable"]` for `/memories/` access: it's the namespace key that isolates each user's persistent memory. Pass `--user <id>` on `muffin analyze` / `muffin screen` to populate it; on LangGraph Platform, set `configurable.user_id` per request (add an `@auth.authenticate` hook once multi-user isolation is required). CLI ships `InMemoryStore`; LangGraph Platform injects managed Postgres automatically.

**Debugging against `agent-chat-ui`** — the UI does not yet populate `configurable.user_id`, so a request without the fallback would raise `MemoryUnavailableError`. For local debugging set `MEMORY_DEBUG_USER_ID=<id>` (or `configurable.memory_debug_user_id`) — this value is consulted by `_memories_namespace` as a fallback before raising, pinning all anonymous traffic to a single `("memories", <id>)` namespace. `docker-compose.yml` passes the env var through to the `langgraph-api` service; just set it in your shell or `.env`. **Do NOT set it in multi-user deployments** — it collapses every request onto a shared namespace.

**Builder example**:

```python
class TickerClassification(AgentState):
    sector: NotRequired[str]
    market: NotRequired[str]
    stock_type: NotRequired[str]

agent = (
    MuffinAgentBuilder(llm, name="criteria_definition")
    .with_system_prompt_template("criteria_definition.jinja")
    .with_sandbox()
    .with_short_term_memory()
    .with_persistent_memory()
    .with_skills(
        ["/skills/valuation/"],
        filter_middleware=SkillFilterMiddleware[TickerClassification](),
    )
    .with_subagents(subagents)
    .with_response_format(AutoStrategy(schema=CriteriaDefinitionOutput))
    .with_context_schema(TickerClassification)
    .build_deep_agent()
)

# Invoke with flat state keys
agent.ainvoke({"messages": [...], "sector": "banking", "market": "developed"})
```

### Design Principles

0. **KISS. Keep it simple stupid**: Implementation has to be simple and extensible, no over-engineering.

1. **Accuracy First, Optimize Later**: Focus development on agent capabilities and accuracy, not compute cost. Optimize based on evaluation metrics.

2. **Structured Outputs Everywhere**: All LLM outputs use Pydantic models with `.with_structured_output()` for type safety.

3. **Self-Hosted & Open**: No vendor lock-in. Supports multiple LLM providers and self-hosted deployment.

4. **Background Processing**: Designed for background analysis. Tolerates rate limits and delays.

5. **Sub-agent Independence**: Each sub-agent is fully independent. You can build your agentic workflows by mixing different sub-agents.

6. **Balance between deterministic and agentic**: When it's possible to do something deterministic - it's done via code. If something requires reasoning or working with unstructured data - it's outsourced to LLM. On top of pre-defined workflow each agent has additional to reason and define evaluation criteria, make tool calls to collect data required for it and evaluate these criteria. Each agent returns set of criteria it's checked, score on each criteria, it's relevance for this specific use-case and reasoning behind selected score and relevance.

7. **Minimize custom code**: Use libraries if exists. e.g. use `backoff` for retry with backoff instead of writing your own. Use `TA-Lib` for technical indicators, etc


## 🛠️ Setup

### Prerequisites

- Python 3.11+
- Docker (required for OpenSandbox sandbox containers; also needed for Docker deployment)
- Node.js (optional, for [MCP inspector](https://github.com/modelcontextprotocol/inspector))

### Step 1: Install Muffin Agent

```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

pip install -e .
```

### Step 2: Set Up OpenBB MCP Server

Muffin Agent uses [OpenBB](https://openbb.co/) as its data backbone via the Model Context Protocol (MCP). The OpenBB MCP server runs separately and must be available at `http://127.0.0.1:8001/mcp`.

> OpenBB has heavy dependencies — we recommend setting it up in a **separate virtual environment** under `extras/openbb/`. See [extras/openbb/README.md](extras/openbb/) for detailed instructions on installation, provider API keys, and startup.

Quick version:

```bash
cd extras/openbb
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
openbb-mcp --port 8001
```

For full OpenBB MCP documentation, see the [official docs](https://docs.openbb.co/odp/python/extensions/interface/openbb-mcp).

### Step 3: Start OpenSandbox

Muffin Agent uses [OpenSandbox](https://github.com/alibaba/OpenSandbox) to execute Python code in isolated containers — ad-hoc calculations, dataframe analysis, and technical indicator computation. Standard financial calculations (ROIC, projections, sensitivity, etc.) run in-process as LangChain tools for faster execution and better observability.

The OpenSandbox server manages container lifecycle. Start it with Docker:

```bash
docker run -d --name opensandbox \
  -p 8080:8080 \
  -v /var/run/docker.sock:/var/run/docker.sock \
  ghcr.io/alibaba/opensandbox/server:latest
```

The server starts on `http://localhost:8080`. No API key is needed for local development.

> **Docker Compose**: When deploying with `docker compose up`, the `opensandbox-server` service starts automatically — no manual step required.

### Step 4: Configure Environment Variables

Copy the example and fill in your values:

```bash
cp .env.example .env
```

| Variable | Required | Description | Where to get it |
|----------|----------|-------------|-----------------|
| `OPENAI_API_KEY` | Yes (OpenAI or OpenRouter via ChatOpenAI) | LLM API key | [OpenAI](https://platform.openai.com/api-keys) or [OpenRouter](https://openrouter.ai/keys) |
| `OPENAI_SITE_URL` | Legacy: only for OpenRouter via `ChatOpenAI` | Base URL override | Set to `https://openrouter.ai/api/v1`. **New deployments** should set `LLM_PROVIDER=openrouter` + `OPENROUTER_API_KEY` instead — `ChatOpenRouter` knows the base URL. |
| `OPENROUTER_API_KEY` | Yes when `LLM_PROVIDER=openrouter` | OpenRouter API key for `ChatOpenRouter` | [OpenRouter](https://openrouter.ai/keys) |
| `ANTHROPIC_API_KEY` | Yes (if using Anthropic) | Anthropic API key | [Anthropic Console](https://console.anthropic.com/settings/keys) |
| `MODEL` | No | Model in `provider/model` format. Used by the legacy single-model `get_llm()` path. | Default: `openai/gpt-oss-120b` (note: no `:free` suffix — free OpenRouter routes are unsafe as production default). Browse [OpenRouter models](https://openrouter.ai/models). |
| `LLM_PROVIDER` | No | `openai`, `anthropic`, or `openrouter` | Default: `openai` |
| `ORCHESTRATOR_MODELS` | No | Comma-separated model chain for orchestrator-role agents (first = primary, rest become `ModelFallbackMiddleware` chain). Each entry passed to `langchain.chat_models.init_chat_model`, so cross-provider chains work (e.g. `anthropic:claude-sonnet-4-6,openrouter:nvidia/nemotron-...:free`). | Default: empty (falls back to `MODEL`) |
| `COLLECTOR_MODELS` | No | Same shape as `ORCHESTRATOR_MODELS`, for data-collection ReAct subagents. | Default: empty |
| `REASONER_MODELS` | No | Same shape as `ORCHESTRATOR_MODELS`, for pure-reasoning agents (validation, valuation, risk). | Default: empty |
| `TEMPERATURE` | No | LLM temperature (0.0–2.0) | Default: `0.1` |
| `LLM_SDK_RETRIES` | No | SDK-level retries for connect-time errors (network, timeouts, 5xx/429 before any response body arrives). Forwarded to `ChatOpenAI`/`ChatAnthropic`/`ChatOpenRouter` as `max_retries=`. Mid-stream errors are retried separately by LangChain's `ModelRetryMiddleware` (hardcoded defaults in `MuffinAgentBuilder`); transient tool errors are retried by `ToolRetryMiddleware`. | Default: `6` |
| `OPENBB_MCP_URL` | No | OpenBB MCP server URL | Default: `http://127.0.0.1:8001/mcp` |
| `OPENSANDBOX_URL` | No | OpenSandbox server address (`host:port`) | Default: `localhost:8080` |
| `OPENSANDBOX_API_KEY` | No | OpenSandbox API key (omit if no auth) | — |
| `OPENSANDBOX_IMAGE` | No | Docker image for sandbox containers | Default: `python:3.11-slim` |
| `SEARXNG_URL` | No | SearxNG base URL | Default: `http://127.0.0.1:8888`. Auto-configured in docker-compose. |
| `FIRECRAWL_MCP_URL` | No | Firecrawl MCP server URL | Default: `http://127.0.0.1:3000/mcp`. Auto-configured in docker-compose. |
| `FIRECRAWL_API_KEY` | No | Firecrawl API key | Default: `local` (any value works with `USE_DB_AUTHENTICATION=false`) |
| `SEARXNG_SECRET_KEY` | No | SearxNG secret key | Required for docker-compose. Set a random string. |
| `LANGFUSE_SECRET_KEY` | No | LLM tracing (optional) | [Langfuse Cloud](https://cloud.langfuse.com) → Settings → API Keys |
| `LANGFUSE_PUBLIC_KEY` | No | LLM tracing (optional) | Same as above |
| `LANGFUSE_BASE_URL` | No | Langfuse host URL | Default: `https://cloud.langfuse.com` |

### Step 5: Verify

```bash
# Check CLI is installed
muffin --help

# Make sure OpenBB MCP server is running (Step 2), then:
muffin price AAPL
```

### Running Tests

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run all tests
pytest

# Run unit tests only
pytest -m unit

# Run with coverage
pytest --cov=muffin_agent tests/

# Run specific test file
pytest tests/test_config.py

# Run integration tests calling APIs
pytest -m live
```

### Code Quality

```bash
# Format code
ruff format src/ tests/

# Lint code
ruff check src/ tests/

# Type check
mypy src/
```

---

## 🖥️ CLI

Muffin ships a `muffin` CLI with subcommands for each agent. Output is streamed in real-time with Rich formatting.

```bash
# Install (registers the `muffin` entry point)
pip install -e .

# Retrieve fundamental data for a ticker
muffin fundamentals AAPL

# Retrieve price data for a ticker
muffin price AAPL

# Retrieve analyst estimates for a ticker
muffin estimates AAPL

# Retrieve options chain for a ticker
muffin options AAPL

# Evaluate a stock (deep agent with subagents)
muffin evaluate AAPL
muffin evaluate AAPL -q "Is this stock undervalued based on fundamentals?"

# Evaluate a single investment criterion
muffin criterion AAPL -c "Does the company have strong and improving profitability?"
muffin criterion MSFT -c "Is the balance sheet healthy?" -q "Focus on debt levels and liquidity"

# Define valuation criteria (with optional pre-classification)
muffin criteria AAPL
muffin criteria JPM --sector banking --market developed --stock-type value

# Custom query
muffin fundamentals MSFT -q "Get income statement and ratios"
muffin price MSFT -q "Get current quote and 1-year historical prices"
muffin estimates MSFT -q "Get analyst price targets and forward PE"
muffin ownership MSFT -q "Get institutional holders and short interest"
muffin news MSFT -q "Get recent news and sentiment"
muffin options MSFT -q "Get options chain and implied volatility surface"
muffin web-search "MSFT AI strategy 2025" --ticker MSFT
muffin web-search "https://ir.microsoft.com/financial-information/annual-reports"

# Help
muffin --help
muffin fundamentals --help
muffin price --help
muffin estimates --help
muffin ownership --help
muffin news --help
muffin options --help
```

**Output features:**
- Real-time token streaming (`stream_mode="messages"`)
- Tool calls shown with yellow labels
- Tool results in Rich panels with syntax-highlighted JSON
- Errors shown in red panels — agent continues gracefully via middleware



## 🚀 Web chat interface

See [docs/deployment.md](docs/deployment.md) for deploying to a LangGraph Standalone Server (Docker + PostgreSQL + Redis).


## 🐞 Debugging locally

Press `F5` on the **LangGraph Dev Server (Debug)** config in VSCode — docker compose starts infra + chat UI, `langgraph dev` runs on the host under debugpy on port 8123, and the UI at http://localhost:3000 routes to it. Breakpoints fire in `src/muffin_agent/**` on every request; edits hot-reload. Full guide: [docs/debugging-locally.md](docs/debugging-locally.md).


## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Built with ❤️ by the Muffin Agent Team**

*Empowering investors with AI-driven analysis*
