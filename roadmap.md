## 📅 Roadmap

### Phase 1

#### Data Collection Agents
- [+] Create example data collection agent
- [+] Develop CLI for agents
- [ ] Create other **High** priority data collection agents from [docs/data-collection-agents.md](docs/data-collection-agents.md):
  - [ ] Equity Estimates agent (8 tools: analyst price targets, consensus, forward EPS/EBITDA/PE/sales)
  - [ ] Equity Ownership & Short Interest agent (9 tools: institutional, insider, 13F, short volume/interest)
  - [ ] Company News agent (2 tools: `news_company`, `news_world`)
- [ ] Integrate new data agents into `create_stock_evaluation_agent` as subagents
- [ ] Add setup guide including guide on getting API keys for OpenBB providers, setting up langfuse and getting other .env variables
  - [ ] Add `.env.example` with all required variables documented
  - [ ] Make OpenBB MCP URL configurable via `OPENBB_MCP_URL` env variable (currently hardcoded to `http://127.0.0.1:8001/mcp` in `config.py`)
- [ ] Handle rate limiting with `openai/gpt-oss-120b:free` model
  - [ ] Add `backoff` to dependencies (design principles say to use it, but it's absent from `pyproject.toml`)


#### Stock Evaluation Agent (v1)
- [+] Developed deep agent that uses data collection agents as sub agents and performs: planning, data collection, data validation, analysis, reflection
- [+] Add logic for data validation that checks data sufficiency, relevance, and temporal correctness
- [ ] Add structured Pydantic output for final evaluation result (score, relevance, reasoning, data_used, data_quality_notes) — currently returns raw markdown, contradicting design principle #2 (Structured Outputs Everywhere)


#### Development & Code Quality
- [ ] Add Claude Code skills for Spec driven development
- [ ] Check Claude Code development via mobile app
- [ ] Fix `pytest-asyncio` missing from `[project.optional-dependencies].dev` in `pyproject.toml` (present only in `[dependency-groups]`)
- [ ] Clarify/implement OpenRouter as a first-class `llm_provider` option — README and `config.py` docstring mention it, but the `Literal["openai", "anthropic"]` type excludes it
- [ ] Remove or defer `TA-Lib` dependency until a Technical Analysis agent is implemented — it is a heavy C extension that is hard to install and currently unused


### Phase 2

#### Data Validation Agent
- [ ] Develop data validation agent that takes criterion and data collected and checks if data is sufficient, data is relevant, if point of time is provided - data is not going beyond that point of time. Add this agent as sub agent to Stock Evaluation Agent. Agent should produce confidence/relevance scores.

#### Criterion evaluation Agent
- [ ] Develop deep agent that takes criterion that needs to be evaluated and with that criterion:
    - defines data needs;
    - calls data collection subagents to collect this data;
    - calls data validation agent to validate the data;
    - evaluates criterion using the data (produces confidence/score/reasoning);
    - reflects on evaluation results and based on reflection results push back on analysis to gather more data or re-evaluate if needed.

#### Criteria evaluation Agent
- [ ] Develop agent that takes as an input ticker and some additional information and:
    - collects data about the ticker (industry, sector, other relevant info);
    - defines list of criteria that needs to be evaluated (with their relevance);
    - calls for each criterion Criterion evaluation subagents;
    - synthesizes results of evaluated criteria and provides final verdict;
    - reflects on evaluation results and based on reflection results push back on analysis to add more criteria or re-synthesize report if needed.

### Valuation agents
- [ ] DCF valuation Agent
- [ ] Explore other valuation methodologies

### Phase 3

#### Agent Evaluations
- [ ] Add support of defining point of time at which data has to be fetched
- [ ] Setup evaluation datasets best on the past stock performances and point of time evaluation
- [ ] Definition evaluation metrics
- [ ] Setup LLM-as-a-judge scoring
- [ ] Setup evaluations with Langfuse
- [ ] Optimize prompts based on evals using langfuse

#### Specialized Agents
- [ ] Integrate tool(s) to get Technical indicators (unblock `TA-Lib` dependency)
- [ ] Develop Specialized Technical Analysis Agent
- [ ] Develop Specialized Fundamental Analysis Agent
- [ ] Develop Specialized Macro economy Analysis Agent
  - [ ] Economy & Macro data collection agent (~38 tools: GDP, CPI, FOMC, FRED series, BLS surveys)
  - [ ] Fixed Income & Rates data collection agent (~22 tools: yield curve, SOFR, treasuries, corporate bonds)
- [ ] Develop Specialized News & Sentiment Agent
- [ ] Develop Specialized Social Networks Agent
- [ ] Develop Specialized Prediction Market Analysis Agent
- [ ] Develop Specialized Strategic & Growth Agent
- [ ] Develop Specialized Competitive Analysis Agent
- [ ] Explore agents from https://github.com/TauricResearch/TradingAgents
- [ ] Explore agents from https://github.com/virattt/ai-hedge-fund

#### Other improvements
- [ ] Save information about past tool call failures in some memory, so later agent can learn from them and avoid doing faulty calls (e.g. if some provider is not setup or some call requires premium subscription)
- [ ] Integrate agent development with langfuse to analyze what changes has to be made based on observations.
- [ ] Add capability to pass pre-defined conditions
- [ ] Think about adding HITL to handle: data fetchnig failure, adjusting instructions, validating criteria, etc
- [ ] Setup the model to generate python code for deterministic functions instead of doing math on it's own.

#### Unbiasing agents
- [ ] When defining data needs for criterion - agent shouldn't know about subagents available, to make sure that data needs are unbaiased
- [ ] When evaluation criterion against data agent shouldn't know about ticker or any other information except data and criterion
- [ ] When reflecting on criterion evaluation - agent shouldn't know about ticker or any other information except data and criterion. It should look only on data provided and criterion evaluation results.
- [ ] When synthesizing results from evaluated criteria - agent shouldn't know about ticker or any other information except criteria evaluation results.

#### CI/CD and testing
- [ ] Add full e2e integreation test mocking LLM calls
- [ ] Add github actions to run integration tests with agents before merging pull requests

### Phase 4
- [ ] Deployment configuration
- [ ] Monitoring & alerting
- [ ] Scale testing
- [ ] Expose agents as API (FastAPI)
- [ ] Expose agents as MCP servers (FastMCP)
- [ ] Developing client app(s)
