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

- **`agents/data_validation.py`** — Pure reasoning agent (no MCP tools) that validates collected data against a criterion. Built with `create_agent(model, system_prompt=...)` and no tools — the ReAct loop resolves to a direct LLM response. Checks sufficiency, relevance, temporal validity, and consistency. Prompt: `data_validation.jinja`. Used as a `CompiledSubAgent` in both stock evaluation and criterion evaluation agents.

- **`agents/subagents.py`** — Shared `build_analysis_subagents(config)` async helper. Creates the standard 14 subagents (13 data collection + 1 validation), each wrapped in `CompiledSubAgent`. Used by both `stock_evaluation.py` and `criterion_evaluation.py`.

- **`agents/criterion_evaluation.py`** — Deep agent that evaluates a single investment criterion. Uses `build_analysis_subagents()` and `create_deep_agent()`. Follows a 5-step workflow: Analyze Criterion → Collect Data → Validate → Evaluate (CoT with dynamic sub-criteria) → Reflect. Produces structured output with score, confidence, signal, reasoning, and counterargument.

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
