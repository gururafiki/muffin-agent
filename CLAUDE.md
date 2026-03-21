# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Muffin Agent is a hierarchical multi-agent stock analysis system built with LangGraph. It orchestrates specialized agents that analyze stocks from multiple perspectives (technical, fundamental, news sentiment, etc.) to produce investment theses with price targets.

## Commands

```bash
# Install
pip install -e .          # runtime deps
pip install -e ".[dev]"   # + dev deps (ruff, mypy, pytest)

# Tests
pytest                           # all tests
pytest -m unit                   # unit tests only
pytest -m live                   # tests hitting live APIs
pytest tests/test_config.py      # single file
pytest --cov=muffin_agent tests/ # with coverage

# Code quality
ruff check src/ tests/           # lint
ruff format src/ tests/          # format
mypy src/                        # type check
```

## Architecture

The source lives in `src/muffin_agent/` and is organized as:

- **`config.py`** — Central `Configuration` Pydantic model. Manages LLM provider selection (OpenAI/Anthropic), API keys, and model parameters. Nodes get config via `Configuration.from_runnable_config(config)` which merges env vars with LangGraph's `RunnableConfig["configurable"]`. Call `configuration.get_llm()` to get a LangChain chat model. Also provides `get_mcp_connections()` for OpenBB MCP server access.

- **`prompts/`** — Jinja2-based prompt templates organized into `data_collection/` and `investment/` subdirectories mirroring the `agents/` structure. Use `render_template(template_name, **kwargs)` from `prompts/__init__.py` with subdirectory-prefixed names (e.g., `render_template("data_collection/equity_fundamentals.jinja")`). Root-level agents (`stock_evaluation`, `criterion_evaluation`, `data_validation`) keep their templates at the prompts root.

- **`utils/observability.py`** — Optional LangFuse tracing via `setup_tracing()`. Returns callback handlers for `RunnableConfig["callbacks"]`. Gracefully degrades if LangFuse is unavailable.

- **`agents/data_collection/`** — Data collection agents using MCP tools with the ReAct pattern. Each agent is a single `.py` file containing:
  - `MCP_TOOLS` — list of allowed MCP tool name strings
  - `create_*_agent(config)` — async; calls shared `get_tools()`, renders the Jinja2 prompt, builds the agent via `create_agent()` from `langchain.agents`

  Shared utilities live in `utils.py`:
  - `get_tools(config, allowed_tools, custom_tools=None)` — loads filtered MCP tools via `MultiServerMCPClient`
  - `handle_tool_errors` — `@wrap_tool_call` middleware that catches tool exceptions

  Currently implemented: `equity_fundamentals.py` (25 tools), `equity_price.py` (5 MCP tools + `execute_python` custom tool via OpenSandbox)

- **`agents/data_validation.py`** — Pure reasoning agent (no MCP tools) that validates collected data against a criterion. Built with `create_agent(model, system_prompt=...)` and no tools — the ReAct loop resolves to a direct LLM response. Checks sufficiency, relevance, temporal validity, and consistency. Prompt: `data_validation.jinja`. Used as a `CompiledSubAgent` in both stock evaluation and criterion evaluation agents.

- **`agents/subagents.py`** — Shared `build_analysis_subagents(config)` async helper. Creates the standard 14 subagents (13 data collection + 1 validation), each wrapped in `CompiledSubAgent`. Used by both `stock_evaluation.py` and `criterion_evaluation.py`.

- **`agents/criterion_evaluation.py`** — Deep agent that evaluates a single investment criterion. Uses `build_analysis_subagents()` and `create_deep_agent()`. Follows a 5-step workflow: Analyze Criterion → Collect Data → Validate → Evaluate (CoT with dynamic sub-criteria) → Reflect. Produces structured output with score, confidence, signal, reasoning, and counterargument.

- **`agents/investment/market_regime.py`** — LangGraph node (Step 2 of investment process) that classifies the current macro/liquidity regime. Uses a **macro-focused subset** of 6 subagents (economy-macro, fixed-income, fama-french, currency-commodities, etf-index + data-validation) built by the private `_build_macro_subagents()` helper — does NOT reuse `build_analysis_subagents()`. Deep agent with `get_backend` sandbox for Python calculations (yield curve slope, factor Z-scores). Supports 3 context modes via `MarketRegimeContext` TypedDict (all fields optional): (a) `ticker` — agent calls `etf_equity_exposure` to derive sector/style, (b) explicit `sector`/`industry`/`country` fields, (c) `query`-only. Output is enforced via `response_format=AutoStrategy(schema=MarketRegimeOutput)` (LLM tool-calling); `market_regime_node` reads `result["structured_response"].model_dump()` — no regex/JSON parsing. Fallback error dict `{"regime_label": "unknown", "error": ..., "raw_output": ...}` if `structured_response` is `None`. Used in both `TickerAnalysisState` (per-ticker) and `ScreeningState` (shared pre-fanout context).

- **`agents/investment/sector_analysis.py`** — LangGraph node (Step 3 of investment process) that assesses sector/industry attractiveness. Uses a **sector-focused subset** of 5 subagents (etf-index, discovery-screening, news, regulatory-filings + data-validation) built by the private `_build_sector_subagents()` helper. Deep agent with `get_backend` sandbox for Python calculations (sector relative performance vs S&P 500, valuation premium/discount, peer return dispersion). Supports 3 context modes via `SectorAnalysisInputState` TypedDict (all fields optional): (a) `ticker` — agent calls `etf_equity_exposure` to derive sector/industry, (b) explicit `sector`/`industry` fields, (c) `query`-only thematic scan. Scores 6 dimensions: cycle position, Porter's Five Forces competitive structure (5 individual force scores + overall attractiveness), thematic drivers (structured list with direction and time_horizon), sector relative valuation, regulatory/legislative backdrop, and alpha opportunity (peer return dispersion). Output is enforced via `response_format=AutoStrategy(schema=SectorViewOutput)`; `sector_analysis_node` reads `result["structured_response"].model_dump()`. Fallback error dict `{"sector": "unknown", "error": ..., "raw_output": ...}` if `structured_response` is `None`. Runs in parallel with `market_regime_node` and `company_analysis_node` (Group 1). Also used as a shared context node in `ScreeningState` before fan-out.

- **`agents/investment/company_analysis.py`** — LangGraph node (Steps 4-5 of investment process) covering Business/Moat/Mgmt/ESG Triage and Financial Quality Deep Dive. Uses a **company-focused subset** of 6 subagents (equity-fundamentals, equity-ownership, regulatory-filings, news, discovery-screening + data-validation) built by the private `_build_company_analysis_subagents()` helper. Deep agent with `get_backend` sandbox for Python calculations (ROIC, FCF conversion, net debt/EBITDA, revenue CAGR 3Y, interest coverage, peer ROIC premium). Supports 2 context modes via `CompanyAnalysisInputState` TypedDict (all fields optional): (a) `ticker` + `query` — standard per-ticker analysis, (b) `query`-only thematic quality screen. Scores 4 dimensions: moat assessment (width/sources/trend/confidence with peer ROIC premium), management quality and capital allocation, ESG and governance triage, and financial quality (margins, ROIC, FCF conversion, leverage). Builds a 5-year financial history time series (parallel arrays + quality narrative) for downstream `forecasting_node` modeling. Issues a triage gate signal (`company_signal`: pass/watch/fail). Output is enforced via `response_format=AutoStrategy(schema=CompanyAnalysisOutput)`; `company_analysis_node` reads `result["structured_response"].model_dump()`. Fallback error dict `{"error": ..., "raw_output": ...}` if `structured_response` is `None`. Runs in parallel with `market_regime_node` and `sector_analysis_node` (Group 1); its output is the primary input for `forecasting_node` and `risk_assessment_node` (Group 2).

- **`agents/investment/forecasting.py`** — LangGraph node (Step 6 of investment process) that builds a 3-year forward financial model with bull/base/bear scenarios anchored to analyst consensus. Uses a **forecasting-focused subset** of 4 subagents (equity-estimates, equity-fundamentals, economy-macro, currency-commodities + data-validation) built by the private `_build_forecasting_subagents()` helper. Deep agent with `get_backend` sandbox for Python scenario modeling (historical calibration via `financial_history` from company_analysis, projection arithmetic for all 3 scenarios, sensitivity table, accruals ratio for earnings quality). Input: `ForecastingInputState` TypedDict (ticker, query, company_analysis, market_regime — all optional). Context modes: (a) full pipeline run (all 4 fields), (b) ticker + query standalone, (c) query-only thematic. Structured output: `ForecastOutput` with nested `Scenario` models (each containing `list[YearlyProjection]` for Y+1/+2/+3 with revenue/EBITDA/EBIT/EPS/FCF), `ConsensusAnchor` (revision_trend_3m, surprise_history, price targets), and `SensitivityDriver` table. Scenario probability anchors keyed to `company_signal` (pass=60/25/15, watch=50/25/25, fail=40/25/35); LLM can deviate with written rationale. EPS set to null if diluted share count unavailable. Output enforced via `response_format=AutoStrategy(schema=ForecastOutput)`; fallback error dict `{"error": ..., "raw_output": ...}` if `structured_response` is `None`. Runs in parallel with `risk_assessment_node` (Group 2); output is primary input for `valuation_node` (Group 3). Writes `forecast` key to state. Full analysis always runs regardless of `company_signal` (useful for both long and short theses).

- **`sandbox/`** — OpenSandbox integration. Sandboxes are discovered/created lazily by `thread_id` metadata — no middleware or state propagation needed.
  - `OpenSandboxBackend` (`backend.py`) — `deepagents.BaseSandbox` implementation backed by a `SandboxSync` container. Pure wrapper; all file operations delegated to shell commands via `execute()`.
  - `SandboxFactory` (`factory.py`, internal) — Discovers running sandboxes by `thread_id` metadata via `SandboxManagerSync.list_sandbox_infos()`. Creates a new sandbox if none found. Works with both `ToolRuntime` and `Runtime` contexts (`thread_id` from `langgraph.config.get_config()`).
  - `get_backend` (`factory.py`) — `BackendFactory` function. Pass as `backend=get_backend` to `create_deep_agent`.
  - `get_sandbox` / `aget_sandbox` (`factory.py`) — Sync/async functions returning a raw sandbox instance for direct use.
  - `execute_python` (`tools.py`) — `@tool` async tool. Discovers the sandbox for the current thread via `aget_sandbox` and executes Python code in it.

## Conventions

- **Ruff** with Google-style docstrings (`D401` imperative mood enforced). `D` and `UP` rules are relaxed in `tests/`.
- **Pydantic** models for all configuration and structured outputs.
- Python 3.11+ required.
- Environment variables for secrets (API keys, LangFuse credentials). See `.env` file.
- MCP (Model Context Protocol) for external data access (OpenBB).
- Use `create_agent` from `langchain.agents` for ReAct agents — do NOT use `create_react_agent` from `langgraph.prebuilt` (deprecated).
- KISS: no empty files or premature abstractions. Extract shared utilities only when patterns emerge across multiple agents.

## Collaboration Preferences

These rules govern how Claude approaches planning, implementation, and communication in this project.

1. **Deep planning first** — Always do deep planning and trade-off evaluation before writing any code. Explore the solution space thoroughly before committing to an approach.

2. **Prefer out-of-the-box solutions** — Before implementing custom logic, research available library features by reading internet documentation and/or library source code. Consider alternative options even if they are not an exact match to the ask. Surface interesting options proactively.

3. **Propose options, don't decide** — When facing a design decision or when multiple approaches exist, present the options and ask for a decision rather than picking one unilaterally. Ask questions before writing substantial code if no existing library/utility has been found — the user may be able to provide documentation pointers.

4. **Explicit approval before implementation** — Always ask for explicit approval before starting implementation. Never exit plan mode unless the user explicitly says to exit or switch mode.

5. **Keep documentation up to date** — After every implementation, update README.md, docs/, roadmap.md, and any other relevant docs as applicable. Add VSCode launch configurations where reasonable. Always include documentation updates as the last step of implementation plans. When trade-offs or tech debt are accepted, document the limitations and add action items to roadmap.

6. **Memorize lessons in CLAUDE.md** — If the user shares information that will be useful in future sessions (e.g. future roadmap tasks, corrections, disagreements, repeating feedback patterns, new constraints), record it in CLAUDE.md. When in plan mode, include the CLAUDE.md memory update as an explicit plan step.
