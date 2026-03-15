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
- [ ] Extend docker compose to allow mounting local code to the code location within docker to allow making changes locally and test them immediately.
- [ ] Add configuration to allow debugging python code when docker compose is launched. Options:
    - using VSCode debugging in docker
    - using `langraph dev`
    - using normal python debugger to execute muffin-cli, just connect it to other services hosted within docker compose.
    - update docker compose to use executed outside of docker compose

### Documentation
- [ ] Document Data Validation agent and add launch.json config
- [x] Document Criterion Evaluation agent and add launch.json config

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

### Criteria evaluation Agent
- [ ] Develop agent that takes as an input ticker and some additional information and:
    - collects data about the ticker (industry, sector, other relevant info);
    - defines list of criteria that needs to be evaluated (with their relevance);
    - calls for each criterion Criterion evaluation subagents;
    - synthesizes results of evaluated criteria and provides final verdict;
    - reflects on evaluation results and based on reflection results push back on analysis to add more criteria or re-synthesize report if needed.

## Valuation agents
- [ ] DCF valuation Agent
- [ ] Explore other valuation methodologies
    - Comparables
    - Precedent txns
    - SOTP

## Core workflow
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
    - Check macro
    - Check sector/industry
    - Understand business and evaluate it
- [ ] Ticker Valuation and forecasting (Steps 5-6 from [docs/investment-process.md](docs/investment-process.md))
    - Do valuations based on fundamentals
    - Create projections and scenarious
- [ ] Relative value (Steps 7 from [docs/investment-process.md](docs/investment-process.md))
    - Peer comparison
    - **TODO**
- **TODO**
- [ ] Analysis check (Steps 6 from [docs/investment-process.md](docs/investment-process.md))

## Phase 3

### Agent Evaluations
- [ ] Add support of defining point of time at which data has to be fetched

### Data collection
- [ ] Iterate over data collection agents, improve prompts based on openbb docs. if needed split to smaller specialized agents.
- [ ] Check https://docs.openbb.co/odp/python/reference . There are a lot of commands that are not covered by MCP.
- [ ] Add fire crawl to collect data from web (consider adding as MCP)

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
- [ ] For structure outputs explore response_format for agents

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
- [ ] Save tool outputs in file system and pass their references for computations instead of generating them within scripts.
- [ ] Explore `context_schema` to store sandbox id/thread id: https://docs.langchain.com/oss/python/langchain/tools#context
- [ ] Keep in memory/readme already written scripts.
- [x] Auto-recreate dead sandboxes — `SandboxFactory` discovers by `thread_id` metadata and creates if not found
- [x] ~~External DB for thread_id→sandbox mapping~~ — solved by OpenSandbox metadata API (`SandboxFilter(metadata={"thread_id": ...})`)
- [ ] Share scripts between agent calls.
- [ ] Once authentication is enabled - store scripts per user in persistent storage and pre-populated sandboxes with them.
- [ ] Think about having separate Coding agent instead of writing scripts within each agent.

### DX
- [ ] Add Claude Code skills for Spec driven development
- [x] Check Claude Code development via mobile app
