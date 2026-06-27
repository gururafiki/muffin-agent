# E2E Integration Testing

This guide describes the **end-to-end integration test harness** for muffin
graphs: how to run a *real* compiled graph while mocking only its external
boundaries, and how to add a test for every new graph by default.

- **Harness code:** [`tests/integration/_harness/`](../tests/integration/_harness)
- **Fixtures:** [`tests/integration/fixtures/`](../tests/integration/fixtures)
- **Worked examples:** [`test_equity_price_collector.py`](../tests/integration/test_equity_price_collector.py),
  [`test_persona_peter_lynch.py`](../tests/integration/test_persona_peter_lynch.py)
- **Enforcement:** [`test_graphs_have_integration_tests.py`](../tests/integration/test_graphs_have_integration_tests.py)

```bash
.venv/bin/pytest tests/integration/ -m integration      # the offline suite
.venv/bin/pytest tests/integration/ -m live             # refresh fixtures (needs docker)
```

## Philosophy: run the real graph, mock only the edges

Existing tests cover two extremes well — pure-function unit tests (tools,
scorers, formatters) and *node-boundary* graph tests that **stub whole ReAct
sub-agents with constant-output functions** (e.g.
`tests/agents/test_trading_decision/test_graph.py` replaces `_add_analyst_nodes`
with `_stub_market`). Integration tests fill the gap in between: they run the
agent's **real** ReAct loop / multi-node graph — real `MuffinAgentBuilder`, real
middleware stack (retry, cache, tool-knowledge), real routing, real deterministic
nodes — and mock **only** the external calls.

> Mock at the **lowest reasonable level**. If something runs deterministically
> with no external dependency (a scoring function, a dedup pass, a reducer,
> rerank math), **do not mock it** — let it run so the test actually exercises it.

### The four seams

| Boundary | Helper | What it does | Kept REAL |
|---|---|---|---|
| **LLM** | `patch_llm(*script)` | A real `ScriptedChatModel` (`BaseChatModel`) replays a scripted timeline of model turns. Patches the single chokepoint `ModelConfiguration.from_runnable_config`. | builder, middleware, `with_structured_output`/`with_fallbacks`/`with_retry`, ReAct loop, routing |
| **MCP tools** | `patch_mcp(scenario=...)` | Fixture-backed `StructuredTool`s; patches `MultiServerMCPClient` so the real `get_tools` name-filter still runs. | `get_tools` filter, `McpConfiguration` |
| **Sandbox** | `patch_sandbox(execute_output=...)` | In-memory `get_backend` (`StateBackend`) + a fake `aget_sandbox` for `execute_python`. | everything except the container |
| **Embeddings** | `patch_embeddings(vectors)` | Canned-vector fake for the research rerank step. | cosine filter, dedup, top-K |

**Never mocked (runs for real):** deterministic compute nodes (`compute_evidence`,
`scoring_helpers`, technicals/sentiment), merge/dedup, rerank math, state reducers
(`operator.add`, `add_messages`), conditional-edge routing, `/scratch/`
`StateBackend`, prompt rendering, structured-output parsing.

## Why a custom `ScriptedChatModel` (not `FakeLLM`)

`langchain.agents.create_agent` (and deep agents, which delegate to it) call
`model.bind_tools(...)`, and the base `BaseChatModel.bind_tools` raises
`NotImplementedError`. The repo's duck-typed `FakeLLM` (in the older node tests)
therefore **cannot drive a real ReAct loop** — it only works for single direct
LLM calls. [`ScriptedChatModel`](../tests/integration/_harness/scripted_model.py)
is a genuine `BaseChatModel` subclass, so the whole real stack runs; only the
bytes the "LLM" emits are scripted.

A **single shared cursor** spans the whole graph: the N-th model call *anywhere*
returns `script[N]`. Authoring helpers:

- `tool_turn(name, args)` — an `AIMessage` with a tool call. Use it for a real
  MCP/sandbox tool call **and** for the final `response_format` turn, where
  `name` is the response schema's class name (e.g. `PeterLynchRawData`).
- `final(text)` — a free-form text answer.
- a bare Pydantic instance — consumed by the direct-call `with_structured_output`
  path (`ModelConfiguration.get_chat_model_for_role(schema=...)`).

The single LLM seam works because `get_chat_model_for_role` resolves models via
`cls.from_runnable_config(config).get_llm_for_role(role)`
([`model_config.py`](../src/muffin_agent/model_config.py)) — so patching
`from_runnable_config` covers both the factory ReAct path and direct-call nodes.

## Recipe 1 — a single ReAct agent

```python
import pytest
from langchain_core.messages import HumanMessage, ToolMessage
from muffin_agent.agents.data_collection.equity_price import (
    create_equity_price_data_collection_agent,
)
from ._harness import final, patch_llm, patch_mcp, patch_sandbox, tool_turn

pytestmark = pytest.mark.asyncio

async def test_quote(config):
    script = (
        tool_turn("equity_price_quote", {"symbol": "AAPL"}),  # ReAct turn 1
        final("AAPL is trading near $201.50."),               # ReAct turn 2
    )
    # patch_sandbox() is required for any agent built `.with_sandbox()` —
    # its FilesystemMiddleware resolves the backend on every model call.
    with patch_mcp("aapl"), patch_sandbox(), patch_llm(*script) as cursor:
        agent = await create_equity_price_data_collection_agent(config)
        result = await agent.ainvoke(
            {"messages": [HumanMessage("Quote AAPL")]}, config=config
        )

    tool_msgs = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert tool_msgs[0].name == "equity_price_quote"
    assert '"last_price": 201.5' in tool_msgs[0].content   # fixture content
    assert cursor.consumed == 2                            # script fully consumed
```

`config` and `store` fixtures come from
[`tests/integration/conftest.py`](../tests/integration/conftest.py).

## Recipe 2 — a multi-node graph with structured output

For a compiled `StateGraph` whose nodes include a ReAct sub-agent with
`response_format=`, a deterministic node, and a direct structured-output LLM node,
script one shared timeline across all of them. See
[`test_persona_peter_lynch.py`](../tests/integration/test_persona_peter_lynch.py).
The deterministic node is **not** mocked — assert it ran by inspecting downstream
state or the rendered prompt (`cursor.last_system_prompt()`).

## Recipe 3 — full tool-allowlist coverage per agent

To prove an agent can execute **every** tool it declares (names resolve through
the real `get_tools` filter, each fixture loads, the loop still completes), batch
all the calls into one turn with `parallel_tool_turn(*(name, args))` — the way a
real LLM batches independent fetches, and cheap on the agent's model-call budget.
[`test_persona_tools_e2e.py`](../tests/integration/test_persona_tools_e2e.py)
parametrises this over all 13 personas: each `collect_data` invokes its whole
`_MCP_TOOLS` list + `execute_python`, asserted via the `ToolMessage`s captured in
`cursor.inputs[1]` (a missing fixture or a tool name drifting from the OpenBB
catalogue fails by name). The council e2e intentionally skips tool fan-out (its
schema-routed model is stateless for parallel personas) — this recipe is the
tool-coverage counterpart.

## Fixtures — one pluggable file per tool

Fixtures live under [`tests/integration/fixtures/`](../tests/integration/fixtures)
as `openbb/<tool>__<scenario>.json` (an OpenBB envelope `{"results": [...], ...}`)
or `firecrawl/<tool>__<scenario>.json` (a JSON list). `load_fixture` returns the
content exactly as the agent sees it — a JSON **string** for OpenBB, a **list**
for Firecrawl. `patch_mcp` builds one fake tool per fixture and the real
`get_tools` selects each agent's allowlist, so **one library serves every graph**.
Add a tool by dropping in a file.

Each fake OpenBB tool advertises the **real** `inputSchema` (`provider`, `symbol`,
…) pulled from the OpenBB catalogue `openbb_mcp_tools.json` (`args_schema_for(name)`
in `_harness/mcp.py`, falling back to a permissive schema for Firecrawl / local
tools, or when the catalogue is absent). The catalogue was moved out of this repo
(commit `c3705a9`) and now ships with the `openbb-mcp-docker` image build; the
harness resolves it via `openbb_catalogue_path()` (`_harness/fixtures.py`), preferring
the `openbb-mcp-docker` sibling submodule in an umbrella checkout and honouring a
legacy `extras/openbb/` copy. In a standalone muffin-agent checkout it is absent, so
`test_generator_covers_every_schematizable_tool` skips. The schema is advertised, not enforced —
`StructuredTool` passes a dict `args_schema` through without validation, so scripted
`tool_turn` args don't need to match it.

**Sourcing is three-tier:**

1. **Hand-authored** (realistic AAPL values) — for tools a test actually *parses*
   (e.g. the deterministic specialists). The ~8 seed fixtures + any you add.
2. **Generated stubs** — every other agent-referenced OpenBB tool (177 of them)
   ships a **schema-correct stub** built from the catalogue `outputSchema` by
   [`_harness/schema_gen.py`](../tests/integration/_harness/schema_gen.py): correct
   field names/types, lightly humanised values. Enough scaffolding for LLM-driven
   collectors (whose tool output the scripted model ignores). Regenerate missing
   stubs (existing files are never overwritten):

   ```bash
   python -c "import sys; sys.path.insert(0,'tests'); \
     from integration._harness.schema_gen import materialize_missing; \
     print(len(materialize_missing()))"
   ```
   A hand-authored fixture simply overrides a stub. `schema_gen.synth_envelope(tool)`
   also backs ad-hoc use.
3. **Live capture** — refresh any fixture to a genuine payload with the MCP stack up:

   ```bash
   docker compose up -d openbb-mcp firecrawl-mcp searxng
   .venv/bin/pytest tests/integration/test_capture_fixtures.py -m live
   ```
   The capture test ([`_harness/capture.py`](../tests/integration/_harness/capture.py))
   self-skips when the MCP stack isn't reachable, so it never breaks a plain run.

## Default workflow: every new graph gets a test

Adding a graph? Add `tests/integration/test_<name>.py` using a recipe above. For a
**deployable** graph (registered in [`langgraph.json`](../langgraph.json)), the
meta-test [`test_graphs_have_integration_tests.py`](../tests/integration/test_graphs_have_integration_tests.py)
**fails** until the graph is either covered (move its id into `COVERED_GRAPHS`) or
explicitly deferred (`PENDING_INTEGRATION_COVERAGE` + a roadmap item). This makes
coverage the default rather than an afterthought.

## Bug found and fixed by this suite ✅

Authoring the persona example exposed a **systemic, pre-existing break** in
compiled-subagent composition that made the **council**, all three
**trading-decision** graphs, and the **persona CLI** non-functional end to end —
invisible because every graph-level test stubs these subagents. The harness
caught it; it is now **fixed** and locked by `test_persona_peter_lynch.py`,
`test_council_graph_e2e.py`, and `test_trading_analysts_e2e.py`. The two
composition rules (now in CLAUDE.md) are:

1. **Input mapping.** A compiled subagent added via `add_node` must receive an
   **explicit field-based `<Name>Input` TypedDict** (`PersonaInput`,
   `AnalystInput`), NOT `agent.input_schema` — `create_agent`'s `.input_schema` is
   a property-less Pydantic `RootModel`, so LangGraph maps `{}` and `_coerce_state`
   raises a `ValidationError` before any model call.
2. **Output propagation.** A field the subagent WRITES and a later node READS must
   be `OmitFromSchema(input=True, output=False)` (kept in the subagent's output).
   `output=True` strips the auto-unpacked value, so the downstream node sees
   nothing.

These are exactly the kind of break a stubbed-node test can't see and an E2E test
catches at the first `.ainvoke` — which is why every new graph ships one.
```
