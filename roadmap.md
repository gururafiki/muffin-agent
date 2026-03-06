# 📅 Roadmap

## Phase 1

### Data Collection Agents
- [x] Create example data collection agent
- [x] Develop CLI for agents
- [x] Add setup guide including guide on getting API keys for OpenBB providers, setting up langfuse and getting other .env variables
- [ ] Create other data collection agents from [docs/data-collection-agents.md](docs/data-collection-agents.md)
    - [x] 1. Equity Fundamentals
    - [x] 2. Equity Price
    - [x] 3. Equity Estimates
    - [x] 4. Equity Ownership & Short Interest
    - [x] 5. Company News
    - [x] 6. Options
    - [x] 7. Economy & Macro
    - [x] 8. Fixed Income & Rates
    - [x] 9. ETF & Index
    - [ ] 10. Discovery & Screening
    - [ ] 11. Currency & Commodities
    - [ ] 12. Regulatory & Filings
    - [ ] 13. Fama-French
- [ ] Validate that all MCP tools (except Utility Tools) are assigned to agents
- [ ] Handle rate limiting with `openai/gpt-oss-120b:free` model


### Stock Evaluation Agent (v1)
- [x] Developed deep agent that uses data collection agents as sub agents and perform:
    - planning;
    - data collect using sub agents;
    - data validation;
    - analyzis;
    - reflect on results
- [x] Add logic for data validation that checks if data is sufficient, data is relevant, if point of time is provided - data is not going beyond that point of time


### Deployment
#### Option 1 (Separate client and agent server):
##### For Server options:
- [x] Setup self-hosted [Standalone Agent Server](https://docs.langchain.com/langsmith/deploy-standalone-server#docker-compose) accept 1M node executions limit for development purpose. Build image with [langgraph cli](https://docs.langchain.com/langsmith/cli#build)
- [-] Setup [aegra](https://github.com/ibbybuilds/aegra)
##### For client:
- [x] Setup client web app. For MVP we can go with [langchain-ai/agent-chat-ui](https://docs.langchain.com/oss/python/langchain/ui)
- [-] Use [LangSmith studio](https://docs.langchain.com/langsmith/studio)
- [-] Use [Agent Chat UI](https://agentchat.vercel.app/)
#### Option 2:
- [ ] Go with [chainlit](https://docs.chainlit.io/integrations/langchain) for both client and server
#### For both options:
- [ ] Make sure that integration with langfuse still works. Probalby requires updating graph compilation to pre-compile callback.

## Phase 2

### Data Validation Agent
- [ ] Develop data validation agent that takes criterion and data collected and checks if data is sufficient, data is relevant, if point of time is provided - data is not going beyond that point of time. Add this agent as sub agent to Stock Evaluation Agent. Agent should produce confidence/relevance scores.

### Criterion evaluation Agent
- [ ] Develop deep agent that takes criterion that needs to be evaluated and with that criterion:
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

## Phase 3

### Agent Evaluations
- [ ] Add support of defining point of time at which data has to be fetched

### Data collection
- [ ] Iterate over data collection agents, improve prompts based on openbb docs. if needed split to smaller specialized agents.
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
- [ ] Save information about past tool call failures in some memory, so later agent can learn from them and avoid doing faulty calls (e.g. if some provider is not setup or some call requires premium subscription)
- [ ] Integrate agent development with langfuse to analyze what changes has to be made based on observations.
- [ ] Add capability to pass pre-defined conditions
- [ ] Think about adding HITL to handle: data fetchnig failure, adjusting instructions, validating criteria, etc
- [ ] Setup the model to generate python code for deterministic functions instead of doing math on it's own.
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

### DX
- [ ] Add Claude Code skills for Spec driven development
- [x] Check Claude Code development via mobile app
