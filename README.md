# 🧁 Muffin Agent

**A hierarchical multi-agent system for comprehensive stock analysis using LangGraph**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2.6+-green.svg)](https://github.com/langchain-ai/langgraph)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🎯 Overview

Muffin Agent is a production-ready, multi-agent stock analysis system that functions as a complete investment research department. Built with LangGraph, it orchestrates specialized agents that analyze stocks from multiple perspectives—technical, fundamental, news sentiment, strategic positioning, and competitive landscape—to produce comprehensive investment theses with price targets.

### Key Features

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

---

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
