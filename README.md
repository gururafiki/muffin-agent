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

Muffin Agent uses [OpenSandbox](https://github.com/alibaba/OpenSandbox) to execute Python code in isolated containers — financial calculations (DCF, WACC), dataframe analysis, and technical indicator computation.

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
| `OPENAI_API_KEY` | Yes (OpenAI or OpenRouter) | LLM API key | [OpenAI](https://platform.openai.com/api-keys) or [OpenRouter](https://openrouter.ai/keys) |
| `OPENAI_SITE_URL` | Only for OpenRouter | Base URL override | Set to `https://openrouter.ai/api/v1` |
| `ANTHROPIC_API_KEY` | Yes (if using Anthropic) | Anthropic API key | [Anthropic Console](https://console.anthropic.com/settings/keys) |
| `MODEL` | No | Model in `provider/model` format | Default: `openai/gpt-oss-120b:free`. Browse [OpenRouter models](https://openrouter.ai/models) |
| `LLM_PROVIDER` | No | `openai` or `anthropic` | Default: `openai` (also used for OpenRouter) |
| `TEMPERATURE` | No | LLM temperature (0.0–2.0) | Default: `0.1` |
| `MAX_CRITERIA` | No | Max evaluation criteria per agent (1–20) | Default: `7` |
| `OPENBB_MCP_URL` | No | OpenBB MCP server URL | Default: `http://127.0.0.1:8001/mcp` |
| `OPENSANDBOX_URL` | No | OpenSandbox server address (`host:port`) | Default: `localhost:8080` |
| `OPENSANDBOX_API_KEY` | No | OpenSandbox API key (omit if no auth) | — |
| `OPENSANDBOX_IMAGE` | No | Docker image for sandbox containers | Default: `python:3.11-slim` |
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

# Custom query
muffin fundamentals MSFT -q "Get income statement and ratios"
muffin price MSFT -q "Get current quote and 1-year historical prices"
muffin estimates MSFT -q "Get analyst price targets and forward PE"
muffin ownership MSFT -q "Get institutional holders and short interest"
muffin news MSFT -q "Get recent news and sentiment"
muffin options MSFT -q "Get options chain and implied volatility surface"

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


## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Built with ❤️ by the Muffin Agent Team**

*Empowering investors with AI-driven analysis*
