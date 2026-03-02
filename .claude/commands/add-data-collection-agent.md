Add a new data collection agent to the muffin-agent project.

Arguments: `$ARGUMENTS`

The first argument is the **agent name** in snake_case (e.g. `macro`).
The remaining arguments are the **MCP tool names** (e.g. `equity_macro_bond equity_macro_cpi`).

If the agent name or MCP tools are missing from the arguments, ask the user for them before proceeding.

---

## Workflow

Make a todo list for all tasks below and work through them one at a time.

Throughout this document, `{name}` refers to the agent name (snake_case) and `{Name}` refers to it in Title Case (e.g. `macro` → `Macro`).

---

### Step 1 — Read context files

Before writing any code, read these files to understand the current state and find the right insertion points:

- `src/muffin_agent/agents/data_collection/__init__.py`
- `src/muffin_agent/agents/stock_evaluation.py`
- `src/muffin_agent/prompts/stock_evaluation.jinja`
- `src/muffin_cli/main.py` (the section just before the `evaluate` command, roughly the last 100 lines)
- `.vscode/launch.json` (the section just before the `Stock Evaluation Agent` configs)
- `README.md` (the data collection agents table)

---

### Step 2 — Create `src/muffin_agent/agents/data_collection/{name}.py`

```python
"""{Name} data collection agent.

ReAct agent that retrieves {description of what the agent does} via OpenBB MCP tools.
"""

from langchain.agents import create_agent

from ...config import Configuration
from ...prompts import render_template
from .utils import ToolErrorHandler, get_tools

MCP_TOOLS = [
    # sorted alphabetically
    "tool_name_1",
    "tool_name_2",
]


async def create_{name}_data_collection_agent(config: Configuration):
    """Build the {name} ReAct agent."""
    tools = await get_tools(config, MCP_TOOLS)
    prompt = render_template("{name}.jinja")
    llm = config.get_llm()
    return create_agent(
        model=llm,
        tools=tools,
        system_prompt=prompt,
        middleware=[ToolErrorHandler()],
    )
```

**Rules:**
- `MCP_TOOLS` must be sorted alphabetically
- Use `create_agent` from `langchain.agents` — **never** `create_react_agent` from `langgraph.prebuilt`
- Always include `middleware=[ToolErrorHandler()]`
- Google-style imperative-mood docstring on the function (e.g. "Build the …")

---

### Step 3 — Create `src/muffin_agent/prompts/{name}.jinja`

Write a plain-text system prompt (no Jinja2 variables needed). Structure:

1. **Role sentence**: "You are a {name} data collection agent. Your role is to …"
2. **Tool listing**: For each tool, one bullet: `- **tool_name**: What it does, when to use it.`
3. **Sequencing note** (if any tools depend on outputs of other tools): Explain the required call order explicitly.
4. **Workflow**: Numbered steps — identify what's needed, call tool(s), summarize findings.
5. **Error handling block** (copy this exactly):

```
IMPORTANT — Error handling rules:
- NEVER retry a tool call with the exact same arguments after it fails. The system will block duplicate failed calls automatically.
- If a tool fails due to missing credentials or unsupported parameters, that tool CANNOT work with these parameters — try different arguments, don't repeat calls with the same arguments.
- Instead of retrying with the same arguments, try: (a) a different tool, (b) different parameters, or (c) report the data as unavailable.
- Do NOT apologize or explain at length when a tool fails. State what failed briefly, then continue with available tools.
```

---

### Step 4 — Update `src/muffin_agent/agents/data_collection/__init__.py`

Add the new import and export. Keep both lists in alphabetical order:

```python
from .{name} import create_{name}_data_collection_agent
```

Add `"create_{name}_data_collection_agent"` to `__all__`.

---

### Step 5 — Update `src/muffin_agent/agents/stock_evaluation.py`

1. Add to the import block from `.data_collection`:
   ```python
   create_{name}_data_collection_agent,
   ```

2. Instantiate the agent inside `create_stock_evaluation_agent`:
   ```python
   {name}_agent = await create_{name}_data_collection_agent(config)
   ```

3. Append a `CompiledSubAgent` entry to the `subagents` list:
   ```python
   CompiledSubAgent(
       name="{name-in-kebab-case}",
       description=(
           "Retrieves {concise description of data, 1-2 sentences, ≤150 chars}."
       ),
       runnable={name}_agent,
   ),
   ```

   The `name` field must be **kebab-case** (e.g. `equity-macro`, not `equity_macro`).

---

### Step 6 — Update `src/muffin_agent/prompts/stock_evaluation.jinja`

1. Increment the subagent count in the opening line (e.g. "six" → "seven").
2. Add a bullet for the new subagent in the same format as the others:
   ```
   - **{name-in-kebab-case}**: {short capability description matching the CompiledSubAgent description}.
   ```

---

### Step 7 — Update `src/muffin_cli/main.py`

Add these two blocks immediately **before** the `_stream_evaluate` function:

```python
async def _stream_{name}(ticker: str, query: str | None) -> None:
    """Build the {name} agent and stream output."""
    from langchain_core.runnables import RunnableConfig

    from muffin_agent.agents.data_collection import create_{name}_data_collection_agent
    from muffin_agent.config import Configuration
    from muffin_agent.utils.observability import setup_tracing

    config = Configuration.from_runnable_config(RunnableConfig(configurable={}))
    callbacks = setup_tracing(session_id=ticker)
    agent = await create_{name}_data_collection_agent(config)

    prompt = (
        f"Ticker: {ticker}. {query}"
        if query
        else f"Get comprehensive {name} data for {ticker}"
    )

    printer = StreamPrinter()
    async for chunk, _metadata in agent.astream(
        {"messages": [HumanMessage(prompt)]},
        config=RunnableConfig(callbacks=callbacks, recursion_limit=40),
        stream_mode="messages",
    ):
        printer.print_chunk(chunk)
    printer.finish()


@app.command()
def {name}(
    ticker: Annotated[str, typer.Argument(help="Stock ticker symbol (e.g. AAPL)")],
    query: Annotated[
        str | None,
        typer.Option("--query", "-q", help="Custom query (overrides default)"),
    ] = None,
) -> None:
    """{One-line description of what this command retrieves}."""
    asyncio.run(_stream_{name}(ticker, query))
```

Note: all imports inside `_stream_{name}` are **local** (inside the function body), matching the existing pattern.

---

### Step 8 — Create `tests/agents/test_{name}.py`

```python
"""Tests for the {name} data collection agent."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from muffin_agent.agents.data_collection.{name} import MCP_TOOLS
from muffin_agent.agents.data_collection.utils import get_tools
from muffin_agent.prompts import render_template


@pytest.mark.unit
class TestMCPTools:
    """Test MCP tool allowlist and filtering."""

    def test_mcp_tools_count(self):
        assert len(MCP_TOOLS) == {expected_count}

    def test_mcp_tools_prefix(self):
        for tool_name in MCP_TOOLS:
            assert tool_name.startswith("{common_prefix}_"), f"Unexpected tool: {tool_name}"

    def test_mcp_tools_sorted(self):
        assert MCP_TOOLS == sorted(MCP_TOOLS)


@pytest.mark.unit
class TestGetTools:
    """Test tool loading and filtering."""

    @pytest.mark.asyncio
    async def test_filters_to_allowed_tools(self):
        mock_tool_allowed = MagicMock()
        mock_tool_allowed.name = MCP_TOOLS[0]

        mock_tool_other = MagicMock()
        mock_tool_other.name = "equity_price_historical"

        mock_client = AsyncMock()
        mock_client.get_tools = AsyncMock(
            return_value=[mock_tool_allowed, mock_tool_other]
        )
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        config = MagicMock()
        config.get_mcp_connections.return_value = {
            "openbb": {
                "url": "http://localhost:8001/mcp",
                "transport": "streamable_http",
            }
        }

        with patch(
            "muffin_agent.agents.data_collection.utils.MultiServerMCPClient",
            return_value=mock_client,
        ):
            tools = await get_tools(config, MCP_TOOLS)

        assert len(tools) == 1
        assert tools[0].name == MCP_TOOLS[0]

    @pytest.mark.asyncio
    async def test_returns_empty_when_no_matching_tools(self):
        mock_tool = MagicMock()
        mock_tool.name = "economy_gdp_real"

        mock_client = AsyncMock()
        mock_client.get_tools = AsyncMock(return_value=[mock_tool])
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        config = MagicMock()
        config.get_mcp_connections.return_value = {}

        with patch(
            "muffin_agent.agents.data_collection.utils.MultiServerMCPClient",
            return_value=mock_client,
        ):
            tools = await get_tools(config, MCP_TOOLS)

        assert tools == []


@pytest.mark.unit
class TestPromptTemplate:
    """Test prompt template rendering."""

    def test_{name}_template_renders(self):
        result = render_template("{name}.jinja")
        assert "{name}" in result.lower()
        assert len(result) > 100
```

If MCP_TOOLS share more than one prefix (e.g. `equity_ownership_` and `equity_shorts_`), replace the single prefix test with one that checks each tool starts with one of the valid prefixes.

---

### Step 9 — Update `tests/agents/test_stock_evaluation.py`

Inside `TestCreateStockEvaluationAgent.test_creates_agent_with_subagents`:

1. Add a mock variable: `mock_{name}_agent = MagicMock()`
2. Add a `patch` context manager:
   ```python
   patch(
       "muffin_agent.agents.stock_evaluation"
       ".create_{name}_data_collection_agent",
       new_callable=AsyncMock,
       return_value=mock_{name}_agent,
   ),
   ```
3. Increment `assert len(subagents) == N` by 1.
4. Append at the end:
   ```python
   assert subagents[N]["name"] == "{name-in-kebab-case}"
   assert subagents[N]["runnable"] is mock_{name}_agent
   ```

---

### Step 10 — Update `.vscode/launch.json`

Insert two new configs immediately **before** the `"Stock Evaluation Agent: AAPL"` entry:

```json
{
    "name": "{Name} Data Collection Agent: AAPL",
    "type": "debugpy",
    "request": "launch",
    "module": "muffin_cli.main",
    "args": [
        "{name}",
        "AAPL"
    ],
    "console": "integratedTerminal",
    "envFile": "${workspaceFolder}/.env",
    "justMyCode": false
},
{
    "name": "{Name} Data Collection Agent: Custom Ticker",
    "type": "debugpy",
    "request": "launch",
    "module": "muffin_cli.main",
    "args": [
        "{name}",
        "${input:ticker}",
        "--query",
        "${input:query}"
    ],
    "console": "integratedTerminal",
    "envFile": "${workspaceFolder}/.env",
    "justMyCode": false
},
```

---

### Step 11 — Update `README.md`

Add a new row to the data collection agents table:

```
| `{name}` | {tool_count} | {short description of data retrieved} |
```

---

### Step 12 — Verify

Run these commands and fix any failures before continuing:

```bash
ruff check src/ tests/
PYTHONPATH=src pytest tests/agents/test_{name}.py -m unit --override-ini="addopts="
PYTHONPATH=src pytest tests/agents/test_stock_evaluation.py -m unit --override-ini="addopts="
```

---

### Step 13 — Commit and push

```bash
git add src/ tests/ .vscode/ README.md
git commit -m "Add {name} data collection agent"
git push -u origin <current-branch>
```

---

### Step 14 — Create pull request

Create a PR using `gh pr create` with:

- **Title**: `Add {name} data collection agent`
- **Body** (use a heredoc):
  - `## Summary` — 2-3 bullets covering: what agent does, tools count, that it's registered as a subagent in `stock_evaluation` and exposed as `muffin {name}` CLI command
  - `## Test plan` — checklist: ruff passes, unit tests pass, `muffin {name} AAPL` streams data (requires OpenBB MCP), `muffin evaluate AAPL` still works with all subagents

Return the PR URL to the user when done.

---

## Wrap up

Provide a summary to the user with:

- Files created and modified
- MCP tools registered
- CLI command added (`muffin {name} AAPL`)
- Verification results (ruff, unit tests)
- PR URL
