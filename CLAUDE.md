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

- **`prompts/`** — Jinja2-based prompt templates. Use `render_template(template_name, **kwargs)` from `prompts/__init__.py` to load and render templates from this directory.

- **`utils/observability.py`** — Optional LangFuse tracing via `setup_tracing()`. Returns callback handlers for `RunnableConfig["callbacks"]`. Gracefully degrades if LangFuse is unavailable.

- **`agents/data_collection/`** — Data collection agents using MCP tools with the ReAct pattern. Each agent is a single `.py` file containing:
  - `MCP_TOOLS` — list of allowed MCP tool name strings
  - `create_*_agent(config)` — async; calls shared `get_tools()`, renders the Jinja2 prompt, builds the agent via `create_agent()` from `langchain.agents`

  Shared utilities live in `utils.py`:
  - `get_tools(config, allowed_tools, custom_tools=None)` — loads filtered MCP tools via `MultiServerMCPClient`
  - `handle_tool_errors` — `@wrap_tool_call` middleware that catches tool exceptions

  Currently implemented: `equity_fundamentals.py` (25 tools), `equity_price.py` (5 MCP tools + `execute_python` custom tool via OpenSandbox)

- **`agents/criterion_evaluation/`** — Scaffolded but not yet implemented.

- **`sandbox/`** — OpenSandbox integration. Two integration points:
  - `OpenSandboxBackend` — `deepagents.BaseSandbox` implementation backed by a `SandboxSync` container. All file operations (read/write/edit/grep/glob) are delegated to shell commands inside the container via `execute()`.
  - `SandboxFactory` — `BackendFactory` callable (`(ToolRuntime) → OpenSandboxBackend`). Stores sandbox IDs per `thread_id`; calls `SandboxSync.connect(id, skip_health_check=True)` on each invocation to reconnect to an existing container, falling back to creating a new one if the container is gone. Pass as `backend=` to `create_deep_agent` for per-conversation isolation.
  - `create_python_execution_tool(config)` — Returns a LangChain async tool. Creates a fresh `Sandbox` per call via `async with`, writes code to a temp file, runs it, deletes the file, then closes the container. Stateless — no cross-call filesystem state.

## Conventions

- **Ruff** with Google-style docstrings (`D401` imperative mood enforced). `D` and `UP` rules are relaxed in `tests/`.
- **Pydantic** models for all configuration and structured outputs.
- Python 3.11+ required.
- Environment variables for secrets (API keys, LangFuse credentials). See `.env` file.
- MCP (Model Context Protocol) for external data access (OpenBB).
- Use `create_agent` from `langchain.agents` for ReAct agents — do NOT use `create_react_agent` from `langgraph.prebuilt` (deprecated).
- KISS: no empty files or premature abstractions. Extract shared utilities only when patterns emerge across multiple agents.
