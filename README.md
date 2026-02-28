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
| `equity_price` | 5 | Current quotes, historical OHLCV, NBBO spreads, price performance, market cap history |

### Design Principles

0. **KISS. Keep it simple stupid**: Implementation has to be simple and extensible, no over-engineering.

1. **Accuracy First, Optimize Later**: Focus development on agent capabilities and accuracy, not compute cost. Optimize based on evaluation metrics.

2. **Structured Outputs Everywhere**: All LLM outputs use Pydantic models with `.with_structured_output()` for type safety.

3. **Self-Hosted & Open**: No vendor lock-in. Supports multiple LLM providers and self-hosted deployment.

4. **Background Processing**: Designed for background analysis. Tolerates rate limits and delays.

5. **Sub-agent Independence**: Each sub-agent is fully independent. You can build your agentic workflows by mixing different sub-agents.

6. **Balance between deterministic and agentic**: When it's possible to do something deterministic - it's done via code. If something requires reasoning or working with unstructured data - it's outsourced to LLM. On top of pre-defined workflow each agent has additional to reason and define evaluation criteria, make tool calls to collect data required for it and evaluate these criteria. Each agent returns set of criteria it's checked, score on each criteria, it's relevance for this specific use-case and reasoning behind selected score and relevance.

7. **Minimize custom code**: Use libraries if exists. e.g. use `backoff` for retry with backoff instead of writing your own. Use `TA-Lib` for technical indicators, etc


## 🖥️ CLI

Muffin ships a `muffin` CLI with subcommands for each agent. Output is streamed in real-time with Rich formatting.

```bash
# Install (registers the `muffin` entry point)
pip install -e .

# Retrieve fundamental data for a ticker
muffin fundamentals AAPL

# Retrieve price data for a ticker
muffin price AAPL

# Custom query
muffin fundamentals MSFT -q "Get income statement and ratios"
muffin price MSFT -q "Get current quote and 1-year historical prices"

# Help
muffin --help
muffin fundamentals --help
muffin price --help
```

**Output features:**
- Real-time token streaming (`stream_mode="messages"`)
- Tool calls shown with yellow labels
- Tool results in Rich panels with syntax-highlighted JSON
- Errors shown in red panels — agent continues gracefully via middleware

## 🛠️ Development


### Installation

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -e .
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


## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Built with ❤️ by the Muffin Agent Team**

*Empowering investors with AI-driven analysis*
