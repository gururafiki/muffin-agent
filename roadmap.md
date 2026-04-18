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
- [ ] Make sure that integration with langfuse still works. Probalby requires updating graph compilation to pre-compile callback.

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
- [ ] Develop agent that takes as an input ticker and some additional information and:
    - collects data about the ticker (industry, sector, other relevant info);
    - defines list of criteria that needs to be evaluated (with their relevance);
    - calls for each criterion Criterion evaluation subagents;
    - synthesizes results of evaluated criteria and provides final verdict;
    - reflects on evaluation results and based on reflection results push back on analysis to add more criteria or re-synthesize report if needed.

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
- [ ] Explore agents from https://github.com/TauricResearch/TradingAgents
- [ ] Explore agents from https://github.com/virattt/ai-hedge-fund

### Other improvements
- [x] Rework tool results cache. Use ToolRuntime.store (share InMemoryStore across all agents). Add tool: `write_cached_tool_output_to_backend` that will write tool output to backend from store for further manipulations. Within middleware check if tool output is already in cache (store) and if it's return it instead of reading it from sandbox. Update prompts to teach them about new tool and workflow to manipulate with data: check cached tool output using `discover_cached_tool_outputs` -> use tool to get output schema -> use `write_cached_tool_output_to_backend` to save output in backend -> execute custom code to manipulate with output.
- [ ] Explore `langchain.agents.middleware.context_editing.ContextEditingMiddleware` and `langchain.agents.middleware.summarization.SummarizationMiddleware`:
    - [ ] Clean failed tools and just summarize what agent shouldn't do based on failure messages
    - [ ] Extract from news important in the current context information only (e.g. extract sentiment, evaluate how article may affect ticket short/long-term, etc)
- [ ] Explore `langchain.agents.middleware.model_retry.ModelRetryMiddleware`. Check if it's applied on top of model retries or not.
- [ ] Utilize jinja capabilities to enrich prompt tempalates with necessary data. I think we should at least include current date.
- [ ] Design work of financial depeartment from investing/trading firm with all the specific workflows they use (heavy webcrawl and reasoning task) and created tailored agents for this.
- [ ] Add citations for the data used when analyzing it (where it comes from, which provider, which command, what period of time covered, fillings, etc)
- [ ] Save information about past tool call failures in some memory, so later agent can learn from them and avoid doing faulty calls (e.g. if some provider is not setup or some call requires premium subscription)
- [ ] Integrate agent development with langfuse to analyze what changes has to be made based on observations.
- [ ] Add capability to pass pre-defined conditions
- [ ] Think about adding HITL to handle: data fetchnig failure, adjusting instructions, validating criteria, etc
- [ ] Agent self-improvement
- [ ] Add an agent to analyze stock price gainers and reason why they have grown to incorporate this knowledge later
- [x] For structure outputs explore response_format for agents — implemented in Market Regime Agent via `AutoStrategy(schema=MarketRegimeOutput)`
- [ ] For cross-run memory - ask user to define it before run, so use defines style they need, depth of analysis, etc and later it's used for all interactions with agent
- [ ] Explore additional Valuation methodologies:
    - [ ] Precedent Transactions — zero coverage. No M&A deal data source exists in OpenBB's MCP tools, so there's no subagent to collect transaction multiples, control premiums, or deal environment context. We'd need an external M&A data provider first. OpenBB doesn't expose M&A deal databases (that's typically Bloomberg/Capital IQ/Refinitiv territory). Without deal data, there's nothing to compute on.
    - [ ] SOTP (Sum-of-the-Parts) — schema field exists (sum_of_parts: dict | None) but is explicitly None / v1 placeholder. Would need segment-level revenue/EBITDA extraction (from 10-K MD&A), per-segment peer multiples, and a new aggregation tool. Needs segment-level financials. The regulatory-filings subagent can fetch 10-K filings, but parsing segment tables from SEC filings is a non-trivial extraction problem.
    - [ ] Residual Income — also mentioned in docs/investment-process.md but not implemented.


### Unbiasing agents
- [ ] When defining data needs for criterion - agent shouldn't know about subagents available, to make sure that data needs are unbaiased
- [ ] When evaluation criterion against data agent shouldn't know about ticker or any other information except data and criterion
- [ ] When reflecting on criterion evaluation - agent shouldn't know about ticker or any other information except data and criterion. It should look only on data provided and criterion evaluation results.
- [ ] When synthesizing results from evaluated criteria - agent shouldn't know about ticker or any other information except criteria evaluation results.


### CI/CD and testing
- [ ] Add full e2e integreation test mocking LLM calls
- [ ] Add github actions to run integration tests with agents before merging pull requests

## Phase 4

### Agent Evaluations
- [ ] Setup evaluation datasets best on the past stock performances and point of time evaluation
- [ ] Definition evaluation metrics
- [ ] Setup LLM-as-a-judge scoring (Explore how to callibrate it)
- [ ] Setup evaluations with Langfuse
- [ ] Optimize prompts based on evals using langfuse

### Deployment
- [ ] Self-hosted infrastructure setup. Use (oracle-cloud-docker-swarm-setup with Dokploy)[https://github.com/gururafiki/oracle-cloud-docker-swarm-setup]
    - [ ] Setup Terraform and Ansible to spin up instances with Docker swarm setup
        - [ ] Spin up independent test and prod swarms
        - [ ] Setup GitHub actions (or Dockploy) to auto deploy to test swarm on merge
    - [ ] Deploy Postgre (or Supabase)
    - [ ] Deploy (langfuse)[https://langfuse.com/self-hosting]
    - [ ] Deploy Agent server, build custom FastAPI/FastMCP wrapper or use paid langsmith plan
    - [ ] Deploy client web app.
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
