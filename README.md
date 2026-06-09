# 🧁 Muffin Agent

**A hierarchical multi-agent system for comprehensive stock analysis using LangGraph**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![LangGraph](https://img.shields.io/badge/LangGraph-1.2+-green.svg)](https://github.com/langchain-ai/langgraph)
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

**Output**: `CriterionEvaluationOutput` enforced via `response_format=AutoStrategy(schema=CriterionEvaluationOutput)` — `criterion_name`, `score`, `confidence` (0.0–1.0), `signal` (5-level Literal), `sub_criteria`, `evidence_summary`, `reasoning`, `counterargument`, `limitations`, `data_sources`. Consumed structurally by the Criteria Analysis Orchestrator.

### Criteria Definition Agent

A standalone deep agent that classifies a ticker by sector, market type (developed/emerging), and stock type (value/growth), then loads matching valuation skills via progressive prompt disclosure to produce sector-specific evaluation criteria with target ranges and methodology guidance.

Uses 5 subagents (etf-index, equity-fundamentals, discovery-screening, economy-macro, data-validation) and 55 valuation skills organised under `skills/valuation/`. Skills are pre-filtered by `SkillFilterMiddleware` based on a flat classification provided as input state — the agent only sees the 4-6 skills relevant to its classification.

**Workflow**: Parse Context → Collect Data (4 subagents in parallel) → Validate → Load Skills & Extract Criteria → Reflect.

**Output**: `CriteriaDefinitionOutput` with ticker, sector, market type, stock type, 5-8 valuation criteria (with target ranges, weights, and guidance), screening questions, and valuation pitfalls.

**CLI**:
```bash
muffin criteria AAPL --sector banking --market developed --stock-type value
```

### Criteria Analysis Orchestrator

A LangGraph orchestrator that runs the full criteria-driven investment pipeline end-to-end: classify the ticker, define and research valuation criteria in parallel, fan each criterion out to the Criterion Evaluation Agent, and synthesise a final view. Wraps the existing Criterion Evaluation and Criteria Definition agents; adds three new agents (ticker classification, valuation-methodology research, synthesis) plus a deterministic merge step.

**Five-stage pipeline:**

| Stage | Node | Type | Purpose |
|-------|------|------|---------|
| 1 | `ticker_classification` | Deep agent (3 data subagents + validation) | Produces `TickerClassificationOutput` (sector / sub_sector / market / stock_type). Short-circuits when CLI flags pre-supply all four flat state keys. |
| 2 | `criteria_definition` | Wraps the existing Criteria Definition Agent | Skill-filtered sector criteria (`CriteriaDefinitionOutput`). Runs in parallel with Stage 3. |
| 3 | `valuation_methodology` | Deep agent (web-search + discovery-screening) | Surfaces the canonical valuation approach plus 2–5 ticker-specific extra criteria the skill stream would miss (`ValuationMethodologyOutput`). Runs in parallel with Stage 2. |
| 4a | `merge_criteria` | Pure Python (no LLM) | Deterministic dedup — canonicalises names, keeps the skill version on tie, re-normalises weights to 1.0, tags `source: "skill" \| "web"`. |
| 4b | `criterion_evaluation` | `Send` fan-out, one Criterion Evaluation Agent per merged criterion | `operator.add` reducer accumulates `CriterionEvaluationOutput` from all parallel workers. Per-criterion concurrency is uncapped. |
| 5 | `synthesis` | Reasoning-only deep agent (no subagents, no tools) | Produces `CriteriaAnalysisSynthesis` — composite score, signal (`strong_sell` … `strong_buy`), weighted breakdown, key positives/negatives, divergences, confidence, thesis paragraph. |

Stages 2 and 3 synchronise on an implicit barrier (same pattern as `investment_analysis`); Stage 4b uses the same `Send` + `operator.add` fan-out pattern as `equity_screening`. Registered in [langgraph.json](langgraph.json) as `criteria_analysis` for Platform autodiscovery.

**CLI**:
```bash
# Auto-classify and run the full pipeline
muffin criteria-analyze AAPL

# Pre-classify to skip Stage 1
muffin criteria-analyze JPM --sector banking --market developed --stock-type value

# With investment mandate
muffin criteria-analyze MSFT -q "Long-only quality bias"
```

### Research Agent

Domain-agnostic Perplexity-style deep research agent. Linear LangGraph pipeline: `classifier → researcher → rerank → writer`. Inspired by [Vane (Perplexica)](https://github.com/ItzCrazyKns/Vane); ships with [classifier flags](src/muffin_agent/agents/research/schemas.py), Vane-defaults rerank (OpenAI-compatible embeddings, cosine ≥ 0.5, top-K 20), and inline `[N]` citation discipline.

**Four-node pipeline:**

| Stage | Node | Type | Purpose |
|-------|------|------|---------|
| 1 | `classifier` | LLM call (collector role, structured output) | Produces `ResearchClassification` — standalone (coref-resolved) query, `task_type` (one of 6), `mode_hint` (speed/balanced/quality), `sources_to_use`, `skip_search`. Routes to writer directly on `skip_search=true`. |
| 2 | `researcher` | Deep agent — `firecrawl_search` + `firecrawl_scrape` (+ caller `extra_tools`) + skills filtered by mode/task_type | Iterative tool-using loop with mode-driven LLM-call budget (speed=2 / balanced=6 / quality=25). Emits `ResearchEvidenceFindings.evidence_chunks`. |
| 3 | `rerank` | Pure Python (no LLM) | Embeds query + chunks via `OpenAIEmbeddings` (provider-configurable), cosine-filters, URL-dedups with content-merge, keeps top-K. |
| 4 | `writer` | LLM call (orchestrator role, structured output) | Produces `ResearchOutput` — markdown answer with inline `[N]` citations, key findings, source list, confidence, missing information, suggested follow-ups. Shape varies per task_type. |

**Pluggable on tools**: callers pass `extra_tools=` (additional `BaseTool` instances — academic / news / finance / internal-docs search) and `extra_sources=` (source names to expose in the classifier's enum). The factory is exposed both as a standalone graph (`build_research_graph`, registered in [langgraph.json](langgraph.json) as `research`) and as a `CompiledSubAgent` (`build_research_subagent`) for embedding inside other agents.

**Skill-driven modes**: `/skills/research/` ships with mode skills (speed/balanced/quality) and task-type skills (research_report / comparison / how_to / summary / debate / factual_qa) plus a universal `citation_discipline` skill. The researcher loads these via `SkillFilterMiddleware[ResearchClassificationFilterState]`, filtered to the current mode + task_type. Add a new skill by dropping a SKILL.md with matching metadata under `/skills/research/`.

**Embeddings** are OpenAI-compatible — works with OpenAI direct (default), OpenRouter (incl. the free `nvidia/llama-nemotron-embed-vl-1b-v2:free` model), vLLM, LM Studio, or Ollama. See `EMBEDDING_BASE_URL` in `.env.example`.

**CLI**:
```bash
# Balanced research with default web sources
muffin research "Latest news on Anthropic Claude 4.7"

# Quality mode for in-depth coverage
muffin research "Postgres vs MySQL for OLTP" --mode quality

# skip_search path — classifier decides no external lookup needed
muffin research "What is 2+2?"

# Explicit task type override
muffin research "How do I set up pgvector?" --task-type how_to --mode quality
```

**As a subagent**:
```python
from muffin_agent.agents.research import build_research_subagent

research = await build_research_subagent(
    config,
    extra_tools=[arxiv_search, news_search],  # optional
    extra_sources=["academic", "news"],       # optional
)
# Pass to MuffinAgentBuilder.with_subagents([research, ...])
```

### Multi-Agent Conference Framework

Generic, reusable subgraph builder for "put N agents with different system prompts in conversation". Use it for debate, peer review, collaborative ideation — anywhere multiple agents take turns producing one message each. Lives at `src/muffin_agent/multi_agent/`; full guide: [docs/multi-agent.md](docs/multi-agent.md).

**Three Participant kinds** (each a `Participant` Protocol implementation — mix freely in one conference):

| Class | LLM cost per turn | Use when |
|---|---|---|
| `LLMParticipant(name, system_prompt_template)` | 1 LLM call; prior conversation rendered as text into the system prompt | Short conferences; existing prompts that already reference `{{ transcript }}` |
| `LLMMessageParticipant(name, system_prompt_template)` | 1 LLM call; prior conversation forwarded as `BaseMessage` thread | Long conferences (prompt-cache reuse on the role-only system prompt) |
| `AgentParticipant(name, agent)` | Full ReAct loop on a compiled muffin agent (tools / sub-agents / middleware available); persistent state across turns via the agent's own per-thread checkpointer | Participant needs tools / skills / memory. **Caller MUST build the agent with `MuffinAgentBuilder(...).with_checkpointer(True)`** (the langgraph sentinel) AND the parent graph or conference MUST be compiled with a real checkpointer instance for per-thread persistence to engage. |

Three other pluggable Protocols: `Moderator` (picks next speaker — `RoundRobinModerator`, `AlternatingModerator`), `Terminator` (decides when to stop — `MaxRoundsTerminator`), optional `Judge` (post-conference synthesis — `StructuredOutputJudge` returns `result.model_dump()` from a Pydantic schema).

**State shape** — one shared `messages: Annotated[list[BaseMessage], add_messages]` reducer is the single source of truth; each turn appends one `AIMessage(content, name=<speaker>)`. Per-agent cursors track the last seen message id so each `AgentParticipant` invocation only receives messages added since it last spoke. The agent's internal tool calls / intermediate AIMessages stay in its own per-thread checkpoint — never leak into parent conference state.

**Worked example**:

```python
from muffin_agent.multi_agent import (
    build_conference_graph, LLMParticipant, AgentParticipant,
    RoundRobinModerator, MaxRoundsTerminator,
)
from langgraph.checkpoint.memory import InMemorySaver

# Optional: an AgentParticipant with tools
bull_agent = (
    MuffinAgentBuilder(model, name="bull")
    .with_system_prompt_template("debate/bull.jinja")
    .with_tool(web_search)
    .with_checkpointer(True)             # ← per-thread persistence sentinel
    .build_react_agent()
)

participants = [
    LLMParticipant("conservative", "debate/conservative.jinja"),
    AgentParticipant("bull", bull_agent),
]
graph = build_conference_graph(
    participants=participants,
    moderator=RoundRobinModerator([p.name for p in participants]),
    terminator=MaxRoundsTerminator(max_rounds=2, num_participants=2),
    checkpointer=InMemorySaver(),         # required when any AgentParticipant is in the lineup
)

result = await graph.ainvoke(
    {},
    config={"configurable": {"thread_id": "debate-001"}},
)
# result["messages"] = [AIMessage(name="conservative", ...), AIMessage(name="bull", ...), ...]
```

**Current production consumer**: `agents/trading_decision/`'s 3-way risk debate (Aggressive / Conservative / Neutral) is composed via `_build_risk_debate_subgraph(max_rounds)` and added as a single `risk_debate` node in the parent graph. The investment debate (Bull/Bear) is still bespoke today; migration is a roadmap item.

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

### Trading Decision Pipeline (`agents/trading_decision/`)

A composable trading-decision pipeline ported from [TauricResearch/TradingAgents](https://github.com/TauricResearch/TradingAgents). **Fully self-contained** — fetches its own data via OpenBB MCP (price / fundamentals / news / ownership) and Firecrawl MCP (social / web) through four compiled analyst ReAct agents that run before the Bull/Bear/Judge/Trader/PM downstream nodes. Does NOT consume outputs from any other muffin pipeline. Full architecture: [docs/trading-decision.md](docs/trading-decision.md).

**Three graph builders** (same `TradingDecisionState`, different depth):

| Builder | Topology | Use when |
|---|---|---|
| `build_investment_debate_graph` | analysts → Bull ↔ Bear → Investment Judge | You only want a structured directional view |
| `build_investment_thesis_graph` | …+ Trader | You want operational entry / stop / sizing too |
| `build_trading_decision_graph` | reflector_resolve → analysts (parallel) → debate → Judge → Trader → risk debate → Portfolio Manager → decision_writeback | Canonical full pipeline with reflection memory |

All three are **async** — each starts by building the four compiled analyst agents (so agent construction cost is amortised to graph-build time, not per call).

**The analyst layer** (4 compiled ReAct agents added directly as parent-graph nodes, running in parallel after `reflector_resolve`):

| Analyst | Tools | Output field |
|---|---|---|
| Market | 4 OpenBB equity_price tools + local `get_indicators` (stockstats over OHLCV) | `market_report` |
| Fundamentals | 7 OpenBB equity_fundamental_* tools | `fundamentals_report` |
| News | `news_company` + `news_world` + `equity_ownership_insider_trading` | `news_report` |
| Social | `news_company` + Firecrawl `firecrawl_search` | `sentiment_report` |

**Adversarial debate, two layers:**

| Stage | Participants | Default rounds | Output |
|---|---|---|---|
| Investment debate | Bull Researcher ↔ Bear Researcher | 2 rounds = 4 turns | `InvestmentJudgeOutput` (5-tier `signal`, conviction, bull/bear cases, catalysts, risks, monitoring checklist) |
| Risk debate | Aggressive → Conservative → Neutral (round-robin) | 1 round = 3 turns | Name-tagged `AIMessage`s in `risk_debate_messages` |

The risk debate is wired through the [Multi-Agent Conference Framework](#multi-agent-conference-framework) — `_build_risk_debate_subgraph(max_rounds)` configures three `LLMParticipant`s + `RoundRobinModerator` + `MaxRoundsTerminator` and adds the result as a single `risk_debate` node in the parent graph. The investment debate (Bull/Bear) still uses bespoke node functions today; migration to the conference framework is a roadmap item.

**Operational translation** — the **Trader** maps the Judge's 5-tier `signal` → 3-tier `action` (sell/hold/buy), entry price, stop loss, take profit, position sizing (anchored to conviction buckets in the prompt), and time horizon (anchored to the Judge's catalysts).

**Final synthesis** — the **Portfolio Manager** consumes the Judge thesis + Trader proposal + `risk_debate_messages` and produces `PortfolioDecisionOutput` — the canonical 5-tier rating (`strong_sell` … `strong_buy`), executive summary, detailed thesis, price target, stop loss, time horizon, position sizing, remaining risks, confidence. May revise the Judge's signal one notch up or down based on what the risk debate surfaced.

**Outcome-driven reflection memory** — decisions are persisted under per-user namespace `("memories", user_id, "decisions")` with key `f"{TICKER}:{YYYY-MM-DD}"`. On the next run, `reflector_resolve_node` walks pending entries: fetches realised returns + alpha vs benchmark (default SPY) via the default `OutcomesFetcher` (pluggable), generates a 2–4 sentence reflection via the Reflector LLM, and injects the most-recent same-ticker + cross-ticker reflections into the next Portfolio Manager prompt. The PM marks `incorporates_past_lessons=true` when it actually cites them. All reflection components degrade silently when the store is unavailable — the pipeline always produces a decision.

**Per-run knobs** (all via `configurable`): `max_investment_debate_rounds` (2), `max_risk_debate_rounds` (1), `reflection_enabled` (`true`), `decision_date` (today UTC), `reflection_holding_days` (5), `reflection_benchmark` (`"SPY"`), `reflection_max_same_ticker` (5), `reflection_max_cross_ticker` (3).

**CLI**:

```bash
# Minimal — the four analysts fetch their own data.
muffin decide AAPL

# With investment mandate
muffin decide AAPL --query "long-term hold candidate"

# Layer caller-provided notes alongside the analyst reports
muffin decide AAPL --narrative "Recent earnings call mentioned X..." --user alice

# Pin decision date + tune debate rounds (deterministic testing)
muffin decide AAPL --decision-date 2026-05-23 --invest-rounds 1 --risk-rounds 1 \
                   --no-reflection
```

Required: `docker compose up -d openbb-mcp firecrawl-mcp searxng` (the four analysts need OpenBB MCP + Firecrawl MCP).

The CLI ships with `InMemoryStore`, so reflection memory persists only within a single Python process today. Wire `PostgresStore` (or run on LangGraph Platform) for cross-session persistence — see [docs/trading-decision.md](docs/trading-decision.md) for the recipe.

### Persona Council (`agents/personas_council/`)

13 famous-investor persona agents (Buffett, Graham, Wood, Munger, Ackman, Burry, Pabrai, Taleb, Lynch, Fisher, Jhunjhunwala, Druckenmiller, Damodaran) ported from [ai-hedge-fund](https://github.com/virattt/ai-hedge-fund) and re-platformed onto muffin's idioms.  Each persona is a **compiled 3-node subgraph** — `collect_data` (a ReAct sub-agent that fetches its own data via curated OpenBB MCP tools) → `compute_evidence` (deterministic Python scoring via [`agents/personas_council/tools/scoring_helpers.py`](src/muffin_agent/agents/personas_council/tools/scoring_helpers.py)) → `render_verdict` (one structured-output LLM call).  Outputs use the 5-tier `InvestmentSignal` (`strong_sell` … `strong_buy`) shared with `trading_decision` and `criteria_analysis`.

The council fans out all 13 personas in parallel — each owns its own MCP data fetch (no shared collection step; calls dedupe through the shared tool-result cache) — then synthesises one consensus rating via an LLM-mediated judge:

```
START → [13 personas in parallel] → council_judge → END
```

It is registered in `langgraph.json` as the `"council"` async graph **factory** (`council_graph.py:build_council_graph`), so LangGraph Platform injects the managed checkpointer / store.

Six **specialist** signal agents emit the same `AnalystSignal` contract: `technicals` (5-strategy ensemble) and `sentiment` (insider + news, 30/70) are fully deterministic; `fundamentals`, `growth`, and `valuation` use a ReAct step only to extract OpenBB fields, then score deterministically; `news_sentiment` is the one LLM specialist (per-headline classification).  Enable them in the council with `--include-specialists` — personas and specialists are wired directly into the graph (no central registry).

**CLI**:

```bash
# Single persona
muffin persona warren_buffett AAPL

# Full 13-persona council + synthesis (add --include-specialists for the 6 specialists)
muffin council AAPL -q "Long-only quality bias"

# Specialists (technicals/sentiment deterministic; news-sentiment uses an LLM)
muffin technicals AAPL
muffin sentiment AAPL
muffin fundamentals-signal AAPL   # bare `fundamentals` is the raw data-collection command
muffin growth AAPL
muffin valuation AAPL
muffin news-sentiment AAPL
```

Full guide: [docs/personas.md](docs/personas.md).

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
| `SUMMARISER_MODEL` | No | Cheap fast model used by `ToolKnowledgeMiddleware` to LLM-summarise tool failures into one-line lessons. When unset, the middleware falls back to deterministic `<tool>: previous call failed — <error>` lesson strings. Recommended: `anthropic/claude-haiku-4-5`. | Default: empty |
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

# Run E2E integration tests (real graphs, mocked LLM/MCP/sandbox — offline)
pytest -m integration

# Run with coverage
pytest --cov=muffin_agent tests/

# Run specific test file
pytest tests/test_config.py

# Run tests calling live APIs (e.g. refresh integration fixtures from MCP)
pytest -m live
```

See [docs/integration-testing.md](docs/integration-testing.md) for the E2E
integration test harness and how to add a test for every new graph.

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

# Run the full criteria-driven analysis pipeline
muffin criteria-analyze AAPL
muffin criteria-analyze JPM --sector banking --market developed --stock-type value
muffin criteria-analyze MSFT -q "Long-only quality bias"

# General-purpose deep research (domain-agnostic, Perplexity-style)
muffin research "Latest news on Anthropic Claude 4.7"
muffin research "Postgres vs MySQL for OLTP" --mode quality
muffin research "How do I set up pgvector?" --task-type how_to --mode quality

# Trading decision pipeline (4 analysts -> Bull/Bear debate -> Judge -> Trader
# -> 3-way risk debate via multi_agent conference -> Portfolio Manager).
# The four analysts fetch their own data via OpenBB MCP + Firecrawl MCP.
muffin decide AAPL
muffin decide AAPL --query "long-term hold candidate"
muffin decide AAPL --narrative "Recent earnings call mentioned X..." --user alice
muffin decide AAPL --decision-date 2026-05-23 --invest-rounds 1 --risk-rounds 1
muffin decide AAPL --no-reflection

# Persona council (ported from ai-hedge-fund). 13 famous-investor personas
# evaluate the same ticker via precomputed deterministic facts + a single
# LLM call each, then a judge synthesises a 5-tier consensus rating.
muffin persona warren_buffett AAPL
muffin council AAPL -q "Long-only quality bias, 5-year horizon"

# Specialist signal agents — deterministic, no LLM (cheap, fast)
muffin technicals AAPL    # 5-strategy technical ensemble
muffin sentiment AAPL     # 30/70 weighted insider + news sentiment

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
