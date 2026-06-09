Implement a new investment process agent in the muffin-agent project.

Arguments: `$ARGUMENTS`

The first argument is the **investment process step number** (2–9, 13–15) from
`docs/investment-process.md`. The second optional argument is the **agent name** in
snake_case (e.g. `sector_analysis`). If the step number is missing, ask for it before
proceeding.

---

## How to use this skill

This skill encodes the full methodology used to build the market regime agent (Step 2).
Study `src/muffin_agent/agents/investment/market_regime.py` and
`src/muffin_agent/prompts/investment/market_regime.jinja` as the canonical implementation reference
before writing any code.

Make a todo list for all tasks below and work through them one at a time. Mark each task
complete as soon as you finish it — do not batch completions.

---

## Step 1 — Deep exploration (do this before any design or code)

Read ALL of these before proceeding:

**Spec and references:**
- `docs/investment-process.md` — target step (What, Who Owns, Inputs/Outputs, Success Criteria)
- Follow at least 3 of the numbered source links cited for that step. Open them with WebFetch
  and extract the key analytical frameworks, dimensions, criteria, or scoring rubrics they describe.

**Existing code:**
- `src/muffin_agent/agents/investment/state.py` — which state keys the step reads and writes
- `src/muffin_agent/agents/investment/{name}.py` — read the stub file and its docstring; treat
  it as a hint only, not a binding spec
- `src/muffin_agent/agents/investment/market_regime.py` — canonical reference implementation
- `src/muffin_agent/agents/subagents.py` — full `CompiledSubAgent` definitions for all 14
  subagents (names, descriptions, and the data collection agents they wrap)
- `src/muffin_agent/agents/data_collection/` — list all agent files; for relevant agents,
  read `MCP_TOOLS` to understand exactly what data is available

**Prompt reference:**
- `src/muffin_agent/prompts/investment/market_regime.jinja` — canonical 5-step prompt structure

---

## Step 2 — Identify analytical dimensions / criteria and present design options

> **Collaboration gate**: Complete this step fully, then **present your findings and at
> least 2–3 design options to the user before writing any code**. Do not proceed to
> Step 3 until the user explicitly approves an approach. Ask follow-up questions if
> anything in the spec is ambiguous — the user may have documentation pointers or
> constraints not visible in the codebase.

Every investment process step evaluates the company/idea through **a set of dimensions or
criteria**. These drive everything else: what data to collect, how to score it, and what
the output schema looks like.

From the spec and its references, identify:

1. **What dimensions or criteria** does this step assess? (e.g., for Step 2: growth cycle,
   inflation, monetary policy, liquidity/risk appetite)
2. **What score or signal** does each dimension produce? (binary pass/fail, 0–1 score,
   labelled classification, etc.)
3. **What is the composite output?** (a single label, a scorecard, a structured memo, etc.)
4. **What are the gates/conditions** under which the idea advances, is flagged, or is blocked?

Write down the dimension list. This becomes the analytical spine of both the prompt and the
output schema.

---

## Step 3 — Design output schema

Design the Pydantic output model **before** looking at what data is available. Work backwards:
"what does the downstream user (PM, next step) need to see?" Use the spec's Inputs/Outputs
section as the primary guide.

**Schema design rules:**
- Nest sub-models for repeated structures (e.g., `DimensionDetail` reused per dimension)
- Use `Literal[...]` for controlled vocabularies (labels, classifications, signals)
- Use `float` for all numeric scores; use `str` for free-text with `Field(description=...)`
- Make ticker-specific or context-specific fields `Optional[...] = None` with a docstring
  explaining when they are populated
- Add `data_sources: list[DataSource] = Field(default_factory=list)` and
  `limitations: list[str] = Field(default_factory=list)` to every top-level output model

**Code structure (leaf models first, root model last):**
```python
class DimensionDetail(BaseModel):
    """Assessment of a single {step} dimension."""
    label: str
    score: float
    direction: str
    key_indicators: str

class {Step}Output(BaseModel):
    """Structured output produced by the {name} deep agent."""
    # ... all fields
    data_sources: list[DataSource] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
```

Propose the schema to the user. Present it as a design option alongside at least one
alternative approach (e.g., flat schema vs. nested, scored vs. classified). Do not proceed
to Step 4 until the user explicitly approves the schema.

---

## Step 4 — Design InputState TypedDict

The InputState serves two purposes: (1) it documents which fields the node reads from the
graph state, and (2) its `__annotations__` keys are used to build the context passed to the
agent.

```python
class {Name}InputState(TypedDict, total=False):
    """{Input state schema for ``{name}_node``.

    Documents which state fields the node reads. All fields optional.
    Context modes: ...
    """
    ticker: str
    query: str
    # add any step-specific fields from state.py (e.g., market_regime, sector_view)
```

**Rules:**
- Use `total=False` **only when all fields are genuinely optional** — i.e., the node
  supports multiple context modes where any combination of inputs can work (as in
  market_regime: ticker-only, sector/industry/country, or query-only). If a field is
  always required (e.g., `ticker` for sector_analysis, upstream step outputs for
  thesis_synthesis), do NOT mark the whole TypedDict `total=False`; instead use a
  mix of required and `Optional` fields (or split into a base + extension)
- Include `ticker` and `query` if the step ever uses them
- Include upstream outputs the step needs (e.g., `market_regime` for step 3+)
- Do NOT include fields the node never touches
- The node will build its context as:
  ```python
  context = {
      k: state[k]  # type: ignore[literal-required]
      for k in {Name}InputState.__annotations__
      if state.get(k)  # type: ignore[union-attr]
  }
  ```
  The format passed to `agent.ainvoke({"input": ...})` can be JSON (`json.dumps(context)`)
  or a structured natural-language description — choose whatever best fits the agent's task.

---

## Step 5 — Select subagents

Choose the **minimal focused subset** of subagents that directly serve this step's data needs.
Do NOT use all 14. The market regime agent uses 5+1 (economy-macro, fixed-income, fama-french,
currency-commodities, etf-index + data-validation).

**Recommended subsets by step** (adjust based on your spec reading):

| Step | Recommended subagents |
|------|----------------------|
| 3. Sector / Industry | etf-index, discovery-screening, news, regulatory-filings, data-validation |
| 4. Business, Moat, Mgmt & ESG | equity-fundamentals, news, regulatory-filings, equity-ownership, data-validation |
| 5. Financial Deep Dive | equity-fundamentals, equity-price, equity-estimates, data-validation |
| 6. Forecasting & Scenarios | equity-fundamentals, equity-estimates, economy-macro, data-validation |
| 7. Valuation & Relative Value | equity-fundamentals, equity-estimates, discovery-screening, etf-index, fixed-income, data-validation |
| 8. Risk & Downside | equity-fundamentals, equity-price, options, fama-french, economy-macro, currency-commodities, data-validation |
| 9. Thesis Synthesis | (synthesis only — use all prior steps' results from state; may not need data collection) |
| 13. Monitoring | equity-fundamentals, equity-price, news, options, data-validation |
| 14. Exit Decision | equity-price, equity-estimates, equity-fundamentals, data-validation |
| 15. Post-Mortem | equity-fundamentals, equity-price, fama-french, economy-macro, data-validation |

**Subagent reuse vs. custom descriptions:**

Before writing a new `CompiledSubAgent(name=..., description=..., runnable=...)`, check
whether the existing definition in `agents/subagents.py` has a description appropriate
as-is for this step.

- **If the standard description serves this agent**: import the data collection agent factory
  (e.g., `create_equity_fundamentals_data_collection_agent`) and create the `CompiledSubAgent`
  with the **same `name` and `description` strings** as in `subagents.py`. Do NOT copy-paste
  the description — copy the exact string from `subagents.py` into your `_build_*_subagents()`
  function.
- **If the description should be tailored** (e.g., to clarify which dimension this subagent
  primarily serves, or to give different fetch guidance): create a new `CompiledSubAgent` with
  a step-specific description. The market regime agent does this — e.g., economy-macro's
  description ends with "Primary source for the growth cycle and inflation dimensions."

Always include `data-validation` as the final subagent.

---

## Step 6 — Design the prompt (5-step workflow)

Study `src/muffin_agent/prompts/investment/market_regime.jinja` closely. Every investment agent prompt
follows this structure:

```
Role sentence (1-2 lines) + grounding rule (NEVER fabricate)

## Available Subagents
| Subagent | Primary Use |  (one row per subagent)

## Workflow
Use write_todos to track progress.

### Step 1 — Parse Context
  - What input fields to expect (list the InputState keys)
  - What dimensions/criteria will be assessed
  - Any conditional logic (e.g., populate ticker-specific output only if "ticker" key present)
  - Write a data collection plan before proceeding

### Step 2 — Collect Data
  - One section per data subagent with specific fetch instructions
  - **Computations — MANDATORY after all subagents return**
    Call dedicated financial tools for standard calculations.
    Use `execute` (sandbox) only for ad-hoc computations not covered by the tools.
    List tool calls and sandbox code blocks with named variable outputs.
    Reference these variable names in Step 4.
  - Error handling: "If a subagent fails, do not retry with identical parameters."

### Validate Data
  - Delegate to data-validation subagent (uses shared `_validation_step.jinja` partial)
  - Act on result: proceed / collect_more_data (at most once) / insufficient_data

### Step 4 — [Classify / Evaluate / Score]
  - For each dimension: Chain-of-Thought reasoning (data points → scale anchors → label + score)
  - Reference the named tool/sandbox outputs from the data collection step
  - **Grounding constraint**: every claim must cite a specific data point with value and source
  - Adverse regime / gate guidance: what to flag when thresholds are breached

### Step 5 — Reflect
  - Internal consistency across dimensions
  - Data recency check (all key indicators within last 3 months?)
  - What would flip the classification?
  - Calibration: is the output appropriately conservative / aggressive given the data?

## Returning Your Analysis
  - "call the structured output tool" (never output raw JSON)
  - Fill in all required fields with format guidance per field
```

**Prompt writing rules:**
- Prompt length: 400–700 words for focused steps; up to 900 for complex multi-dimension steps
- Use dedicated financial tools (from `tools/`) for standard calculations; use `execute` (sandbox) only for ad-hoc computations not covered by the tools. Never perform arithmetic in reasoning text.
- The ticker-conditional logic belongs in Step 1 and Step 4 of the prompt, not in Python code
- Apply guardrails from `.claude/skills/financial-prompt-engineering/` (hallucination prevention,
  temporal anchoring, no-fabrication clause, confidence calibration, data degradation)
- For multi-dimensional scoring: use explicit scale anchors (1.0 = X, 0.5 = Y, 0.0 = Z)

---

## Step 7 — Validate prompt against the spec

After drafting the prompt, perform a mandatory spec-compliance check:

Re-read `docs/investment-process.md` for the target step AND the source links you fetched in
Step 1. Verify the prompt covers ALL of:

| Spec element | Check |
|---|---|
| **Inputs** the spec says this step receives | Each input has a corresponding subagent fetch instruction |
| **Outputs** the spec says this step produces | Each output field exists in the Pydantic schema |
| **Success Criteria / Gates** | The prompt's Step 4 or Step 5 explicitly addresses the gate conditions |
| **Role context** (who owns this step) | The role sentence frames the analytical perspective correctly |
| **Cross-references from source links** | Key frameworks from the cited papers/articles are embedded in dimension definitions and scale anchors |

Fix any gaps before proceeding.

> **Implementation gate**: After completing Steps 2–7 (dimensions, schema, InputState,
> subagents, prompt design, spec validation), **present a complete design summary to the
> user** — dimensions identified, output schema, subagents selected, and the prompt outline
> — and ask for explicit approval before writing any Python or Jinja2 files. Do not proceed
> to Step 8 until the user says to go ahead.

---

## Step 8 — Create `src/muffin_agent/prompts/investment/{name}.jinja`

Write the prompt from Step 6 into a Jinja2 template file. No Jinja2 variables needed — this
is a static system prompt.

---

## Step 9 — Create `src/muffin_agent/agents/investment/{name}.py`

Use this exact file structure (section markers are mandatory for consistency):

```python
"""Stage {N}: {Title}."""

import json
from typing import Any, Literal

from deepagents import CompiledSubAgent, create_deep_agent
from langchain.agents.structured_output import AutoStrategy
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from muffin_agent.agents.data_collection import (
    create_{agent1}_data_collection_agent,
    # ...
)
from muffin_agent.agents.data_validation import create_data_validation_agent
from muffin_agent.model_config import ModelConfiguration
from muffin_agent.prompts import render_template
from muffin_agent.sandbox import get_backend

# ── Input state schema ─────────────────────────────────────────────────────────


class {Name}InputState(TypedDict, total=False):
    """Input state schema for ``{name}_node``."""
    ticker: str
    query: str
    # upstream outputs if needed


# ── Output schema ─────────────────────────────────────────────────────────────


class DimensionDetail(BaseModel):
    # ... leaf models first, root model last


class {Name}Output(BaseModel):
    """Structured output produced by the {name} deep agent."""
    # ...
    data_sources: list[DataSource] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)


# ── Subagent builder ──────────────────────────────────────────────────────────


async def _build_{name}_subagents(config: RunnableConfig) -> list[CompiledSubAgent]:
    """Build the focused subagent set for {name} analysis.

    Return {N} data collection subagents + 1 data validation subagent.
    Excludes subagents whose data is not relevant to Step {N}.
    """
    agent1 = await create_{agent1}_data_collection_agent(config)
    # ...
    validation_agent = await create_data_validation_agent(config)

    return [
        CompiledSubAgent(
            name="{subagent-kebab-name}",
            description=("..."),
            runnable=agent1,
        ),
        # ...
        CompiledSubAgent(
            name="data-validation",
            description=(
                "Validates collected data against a criterion. Checks "
                "sufficiency, relevance, temporal validity, and consistency. "
                "Returns per-dimension scores (0-1), overall confidence/"
                "relevance scores, identified gaps, and a recommendation "
                "(proceed/collect_more_data/insufficient_data). Use after "
                "data collection, before analysis. Pass the criterion, "
                "analysis date, and all collected data in the task instruction."
            ),
            runnable=validation_agent,
        ),
    ]


# ── Agent factory ─────────────────────────────────────────────────────────────


from muffin_agent.tools.{domain} import (
    {tool_1},
    {tool_2},
    # ... import only tools this agent needs
)


async def create_{name}_agent(config: ModelConfiguration, store: BaseStore | None = None):
    """Build the {name} deep agent.

    Create a deep agent that [what it does in 1-2 sentences].
    """
    subagents = await _build_{name}_subagents(config)
    prompt = render_template("investment/{name}.jinja")
    llm = config.get_llm()

    return create_deep_agent(
        model=llm,
        system_prompt=prompt,
        subagents=subagents,
        tools=[
            {tool_1},
            {tool_2},
            # ... only tools relevant to this agent
        ],
        backend=get_backend,
        store=store,
        response_format=AutoStrategy(schema={Name}Output),
    )


# ── Node ──────────────────────────────────────────────────────────────────────


from muffin_agent.agents.investment.utils import run_deep_agent_node


async def {name}_node(
    state: {Name}InputState, config: RunnableConfig
) -> dict[str, Any]:
    """Stage {N}: {Title}.

    [Node docstring: what it does, parallel/sequential position in the graph,
    which state keys it reads ({Name}InputState), which state key it writes.]
    """
    return await run_deep_agent_node(
        state=state,
        config=config,
        agent_factory=create_{name}_agent,
        input_state_type={Name}InputState,
        state_key="{state_key}",
        error_fallback={...},  # step-specific fallback fields
    )
```

**Implementation rules:**
- Use `create_deep_agent`, not `create_react_agent` or `create_agent`
- Always pass `tools=[...]` with the relevant financial tools from `muffin_agent.tools`
- Always pass `backend=get_backend` — the sandbox is needed for ad-hoc computations (financial history arrays, custom logic)
- Always pass `response_format=AutoStrategy(schema={Name}Output)`
- Use `run_deep_agent_node()` from `muffin_agent.agents.investment.utils` instead of inlining the node pattern — it handles context building, invocation, structured output extraction, and error fallback with exception logging
- The error fallback dict must include at minimum `"error"` and `"raw_output"` keys
- The state key written (`{state_key}`) must match the field name in `state.py`
- Section separator comments `# ── Section ──...` must span to column 79

---

## Step 10 — Create `tests/agents/test_{name}.py`

All test classes use `@pytest.mark.unit`. Write 6 test classes:

```python
"""Tests for the {name} investment agent."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError

from muffin_agent.agents.investment.{name} import (
    {Name}InputState,
    {Name}Output,
    create_{name}_agent,
    {name}_node,
)
from muffin_agent.prompts import render_template


@pytest.mark.unit
class TestPromptTemplate:
    """Verify prompt renders and contains required structural elements."""

    def test_renders(self):
        result = render_template("investment/{name}.jinja")
        assert len(result) > 200

    def test_contains_subagent_table(self):
        result = render_template("investment/{name}.jinja")
        for name in ["{subagent-1}", "{subagent-2}", "data-validation"]:
            assert name in result

    def test_contains_workflow_steps(self):
        result = render_template("investment/{name}.jinja")
        for step in ["Step 1", "Step 2", "Validate Data", "Step 4", "Step 5"]:
            assert step in result

    def test_contains_output_schema_keys(self):
        result = render_template("investment/{name}.jinja")
        # Check the key output field names are mentioned in the "Returning" section
        for field in ["{key_output_field_1}", "{key_output_field_2}"]:
            assert field in result

    def test_grounding_constraint_present(self):
        result = render_template("investment/{name}.jinja")
        assert "NEVER" in result  # no-fabrication clause

    def test_computation_mandatory_marker(self):
        result = render_template("investment/{name}.jinja")
        assert "MANDATORY" in result  # computation tools marker

    def test_reflection_step_present(self):
        result = render_template("investment/{name}.jinja")
        assert "consistency" in result.lower() or "reflect" in result.lower()

    # Add step-specific tests for gates, adverse conditions, etc.


@pytest.mark.unit
class TestInputState:
    """Verify InputState TypedDict structure."""

    def test_all_fields_optional(self):
        # TypedDict with total=False: instantiation with no fields works
        state: {Name}InputState = {}
        assert state == {}

    def test_annotations_contain_expected_keys(self):
        keys = set({Name}InputState.__annotations__)
        assert "ticker" in keys
        assert "query" in keys
        # add step-specific keys


@pytest.mark.unit
class TestOutputModel:
    """Verify Pydantic output schema validation."""

    def test_valid_full_instance(self):
        output = {Name}Output(
            # ... provide all required fields
        )
        assert output is not None

    def test_optional_fields_absent_when_none(self):
        output = {Name}Output(
            # ... required fields only
        )
        dumped = output.model_dump(exclude_none=True)
        assert "{optional_field}" not in dumped

    def test_invalid_literal_raises(self):
        with pytest.raises(ValidationError):
            {Name}Output(
                # ... required fields with an invalid Literal value
            )

    def test_model_dump_serializable(self):
        output = {Name}Output(
            # ... required fields
        )
        dumped = output.model_dump()
        assert isinstance(dumped, dict)


@pytest.mark.unit
class TestNodeJsonInput:
    """Verify the node serializes state context correctly."""

    @pytest.mark.asyncio
    async def test_passes_ticker_and_query_in_input(self):
        mock_agent = AsyncMock()
        mock_agent.ainvoke = AsyncMock(return_value={"structured_response": None})

        with (
            patch(
                "muffin_agent.agents.investment.{name}.create_{name}_agent",
                return_value=mock_agent,
            ),
            patch(
                "muffin_agent.agents.investment.utils"
                ".ModelConfiguration.from_runnable_config",
                return_value=MagicMock(),
            ),
        ):
            await {name}_node(
                {"ticker": "AAPL", "query": "AI infrastructure"}, MagicMock()
            )

        call_args = mock_agent.ainvoke.call_args
        raw_input = call_args[0][0]["input"]
        parsed = json.loads(raw_input)
        assert parsed["ticker"] == "AAPL"
        assert parsed["query"] == "AI infrastructure"

    @pytest.mark.asyncio
    async def test_omits_missing_state_fields(self):
        mock_agent = AsyncMock()
        mock_agent.ainvoke = AsyncMock(return_value={"structured_response": None})

        with (
            patch(
                "muffin_agent.agents.investment.{name}.create_{name}_agent",
                return_value=mock_agent,
            ),
            patch(
                "muffin_agent.agents.investment.utils"
                ".ModelConfiguration.from_runnable_config",
                return_value=MagicMock(),
            ),
        ):
            await {name}_node({"query": "tech sector"}, MagicMock())

        raw_input = mock_agent.ainvoke.call_args[0][0]["input"]
        parsed = json.loads(raw_input)
        assert "ticker" not in parsed


@pytest.mark.unit
class TestCreateAgent:
    """Verify agent factory wires subagents and response_format correctly."""

    @pytest.mark.asyncio
    async def test_creates_correct_subagent_count(self):
        # Mock all data collection agent factories
        mock_subagent_factories = [
            patch(
                f"muffin_agent.agents.investment.{name}"
                f".create_{dc_agent}_data_collection_agent",
                new_callable=AsyncMock,
                return_value=MagicMock(),
            )
            for dc_agent in ["{dc_agent_1}", "{dc_agent_2}"]  # list all
        ]
        mock_validation = patch(
            "muffin_agent.agents.investment.{name}.create_data_validation_agent",
            new_callable=AsyncMock,
            return_value=MagicMock(),
        )
        mock_create_deep = patch(
            "muffin_agent.agents.investment.{name}.create_deep_agent",
            return_value=MagicMock(),
        )

        config = MagicMock()
        config.get_llm.return_value = MagicMock()

        with mock_create_deep as mock_deep, mock_validation, *mock_subagent_factories:
            await create_{name}_agent(config)
            call_kwargs = mock_deep.call_args[1]
            assert len(call_kwargs["subagents"]) == {expected_count}  # N data + 1 validation

    @pytest.mark.asyncio
    async def test_passes_get_backend(self):
        # ... verify backend=get_backend is passed to create_deep_agent

    @pytest.mark.asyncio
    async def test_uses_auto_strategy_response_format(self):
        # ... verify response_format is AutoStrategy(schema={Name}Output)


@pytest.mark.unit
class TestNode:
    """Verify node behavior: output key, error fallback, structured response."""

    @pytest.mark.asyncio
    async def test_returns_correct_state_key(self):
        mock_output = MagicMock()
        mock_output.model_dump.return_value = {"key": "value"}

        mock_agent = AsyncMock()
        mock_agent.ainvoke = AsyncMock(
            return_value={"structured_response": mock_output}
        )

        with (
            patch(
                "muffin_agent.agents.investment.{name}.create_{name}_agent",
                return_value=mock_agent,
            ),
            patch(
                "muffin_agent.agents.investment.utils"
                ".ModelConfiguration.from_runnable_config",
                return_value=MagicMock(),
            ),
        ):
            result = await {name}_node({"query": "test"}, MagicMock())

        assert "{state_key}" in result

    @pytest.mark.asyncio
    async def test_error_fallback_when_no_structured_response(self):
        mock_agent = AsyncMock()
        mock_agent.ainvoke = AsyncMock(
            return_value={"structured_response": None, "output": "raw text"}
        )

        with (
            patch(
                "muffin_agent.agents.investment.{name}.create_{name}_agent",
                return_value=mock_agent,
            ),
            patch(
                "muffin_agent.agents.investment.utils"
                ".ModelConfiguration.from_runnable_config",
                return_value=MagicMock(),
            ),
        ):
            result = await {name}_node({"query": "test"}, MagicMock())

        payload = result["{state_key}"]
        assert "error" in payload
        assert payload["raw_output"] == "raw text"
```

---

## Step 11 — Verify `state.py` and `__init__.py`

Check that:
1. `src/muffin_agent/agents/investment/state.py` has a field matching `{state_key}` in
   both `TickerAnalysisState` and/or `ScreeningState` (whichever applies)
2. `src/muffin_agent/agents/investment/__init__.py` already exports `{name}_node`
   (it was pre-added from the stub); if not, add it

---

## Step 12 — Update CLAUDE.md

Add an entry to the `agents/investment/` section of CLAUDE.md describing the new agent:

```
- **`agents/investment/{name}.py`** — [Step N description: what the agent does,
  which subagents it uses, what context modes it supports (ticker/query/explicit fields),
  how structured output is enforced, what state key it writes]
```

---

## Step 13 — Update README.md

Add a row or section to the investment workflow agents table (or create it if absent):

```
| Step {N} | {Name} | {subagent count} data subagents + validation | {one-line description} |
```

---

## Step 14 — Update roadmap.md (if applicable)

If `roadmap.md` exists and has a todo item for this step, mark it as done.

---

## Step 15 — Verify

Run these commands and fix all failures before committing:

```bash
ruff check src/ tests/
ruff format src/ tests/
PYTHONPATH=src pytest tests/agents/test_{name}.py -m unit --override-ini="addopts=" -v
```

---

## Step 15b — Add an E2E integration test (required)

Add `tests/integration/test_{name}.py` following the multi-node recipe in
[docs/integration-testing.md](../../docs/integration-testing.md) and the worked
example `tests/integration/test_persona_peter_lynch.py`. Build the real node/graph
via its factory and mock only the boundaries:

- `patch_llm(...)` — one shared script across the node's subagent calls. For a
  deep-agent node with `response_format=`, the final ReAct turn is
  `tool_turn("<OutputSchemaClassName>", {...})`; a direct `get_chat_model_for_role`
  call consumes a bare Pydantic instance.
- `patch_mcp("aapl")` for subagent MCP tools; `patch_sandbox()` if the node uses
  `get_backend`/`.with_sandbox()`/`execute_python`.
- Let deterministic compute run for real; assert on the structured output and
  `cursor.consumed`.

Verify offline: `.venv/bin/pytest tests/integration/ -m integration`. (Note: if the
node composes a compiled subagent via `add_node(..., input_schema=agent.input_schema)`,
see the known composition bug in [docs/integration-testing.md](../../docs/integration-testing.md)
— such e2e tests are `xfail` until that is fixed.)

## Step 16 — Commit and push

```bash
git add src/ tests/ README.md CLAUDE.md
git commit -m "Add {name} investment agent (Step {N})"
git push -u origin <current-branch>
```

---

## Wrap up

Provide a summary to the user with:

- Files created and modified
- Pydantic output model root class and key dimensions/criteria
- Subagents selected and count
- Prompt length (approximate word count)
- Verification results (ruff, unit tests pass count)
- Any limitations or design trade-offs accepted
