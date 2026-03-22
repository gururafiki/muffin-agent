# Cross-Run Memory

## Problem

Each investment analysis run is stateless. The agent cannot recall observations from prior analyses — patterns like "FRED CPI data consistently lags by 2 weeks" or "AAPL Services segment has beaten consensus 4 quarters in a row" must be rediscovered every time.

This limits the agent's ability to calibrate its analysis, anticipate data gaps, or incorporate institutional knowledge that accumulates over multiple analysis runs.

## Solution

A shared `AGENTS.md` memory file injected into all 4 investment agent prompts via Jinja2 template variable. This is **Option B: read-only** — the simplest safe approach where:

- Agents **read** accumulated observations at the start of each run
- Entries are **added manually** (or via a future write-back mechanism)
- No write-back from agents — avoids hallucinated or low-quality memory entries

## How It Works

### Memory file

`src/muffin_agent/agents/investment/AGENTS.md` contains three sections:

```markdown
# Investment Agent Memory

## Observations
<!-- Append per-ticker observations below. Format: YYYY-MM-DD | TICKER | observation -->

## Sector Trends
<!-- Append sector-level observations. Format: YYYY-MM-DD | SECTOR | trend -->

## Model Calibration
<!-- Notes on model accuracy, systematic biases, recurring data gaps -->
```

### Memory loading

`load_agent_memory()` in `utils.py` reads the file and applies a smart skip:

- If the file is missing → returns empty string
- If the file contains only the seed template (no real entries) → returns empty string
- If any section has real content (not just HTML comment placeholders) → returns the full file content

This prevents injecting a useless seed template into the prompt.

```python
def load_agent_memory() -> str:
    try:
        content = _AGENTS_MD.read_text()
    except FileNotFoundError:
        return ""

    # Check each section for real content beyond seed placeholders
    for section in ("## Observations", "## Sector Trends", "## Model Calibration"):
        # ... scan for lines that aren't seed template text
        if has_real_content:
            return content
    return ""
```

### Template injection

All 4 investment agent factory functions pass memory to `render_template()`:

```python
memory = load_agent_memory()
prompt = render_template("investment/market_regime.jinja", memory=memory)
```

A shared Jinja2 partial (`_memory.jinja`) conditionally renders the memory section:

```jinja
{% if memory %}
## Prior Observations (Cross-Run Memory)

The following observations were recorded from prior analysis runs. Use them to
calibrate your analysis — e.g., known data gaps, sector-specific patterns, or
model biases. Treat these as context, not constraints: verify against current
data before relying on them.

{{ memory }}
{% endif %}
```

The partial is included in all 4 investment templates via `{% include 'investment/_memory.jinja' %}`.

### Adding memory entries

Manually edit `AGENTS.md` to add observations:

```markdown
## Observations
2026-03-15 | AAPL | Services revenue consistently beats consensus; weight Services segment higher in forecasting
2026-03-18 | TSLA | Automotive margins volatile Q-over-Q; use trailing 4Q average, not latest quarter

## Sector Trends
2026-03-10 | TECH | AI infrastructure capex cycle accelerating; hyperscaler capex guidance up 30%+ YoY

## Model Calibration
FRED CPI data typically lags by 2 weeks after release date — use BLS direct feed if available
Fama-French factor data from Ken French site updates monthly with ~3 week lag
```

## Files Changed

| File | Change |
|------|--------|
| `src/muffin_agent/agents/investment/AGENTS.md` | New: seed memory file with 3 sections |
| `src/muffin_agent/agents/investment/utils.py` | `load_agent_memory()` function and `_AGENTS_MD` path constant |
| `src/muffin_agent/prompts/investment/_memory.jinja` | New: shared partial for conditional memory injection |
| `src/muffin_agent/prompts/investment/market_regime.jinja` | `{% include 'investment/_memory.jinja' %}` |
| `src/muffin_agent/prompts/investment/sector_analysis.jinja` | Same |
| `src/muffin_agent/prompts/investment/company_analysis.jinja` | Same |
| `src/muffin_agent/prompts/investment/forecasting.jinja` | Same |
| `src/muffin_agent/agents/investment/market_regime.py` | Pass `memory=load_agent_memory()` to `render_template()` |
| `src/muffin_agent/agents/investment/sector_analysis.py` | Same |
| `src/muffin_agent/agents/investment/company_analysis.py` | Same |
| `src/muffin_agent/agents/investment/forecasting.py` | Same |
| `tests/agents/test_memory.py` | 14 tests: seed detection, file missing, real content loading, template integration |

## Configuration

No runtime configuration. Memory is controlled entirely by the content of `AGENTS.md`.

| State | Behaviour |
|-------|-----------|
| File missing | No memory injected (graceful degradation) |
| File has only seed template | No memory injected (avoids useless boilerplate) |
| File has real entries | Full file content injected into all 4 agent prompts |

## Verification

```bash
pytest tests/agents/test_memory.py -v
```

Tests verify:
- Real entries in any section trigger memory injection
- Seed-only file returns empty string
- Missing file returns empty string
- All 4 templates accept and render the `memory` variable
- Memory section is omitted when `memory=""` or not passed

## Limitations & Future Work

- **Read-only (Option B)**: Agents cannot write back observations. This is intentional for the pilot — prevents hallucinated or low-quality entries from accumulating. Entries must be added manually.
- **Option A (read-write)**: A future upgrade would use `CompositeBackend(routes={"/memory/": FilesystemBackend(...)})` to let agents update `AGENTS.md` via the built-in `edit_file` tool. This requires quality controls (e.g., review/approval of agent-written entries) before enabling.
- **Single shared file**: All 4 agents read the same `AGENTS.md`. A future improvement could split into per-agent memory files (e.g., `market_regime_memory.md`, `company_analysis_memory.md`) for more targeted context.
- **No TTL or expiry**: Old entries remain indefinitely. Manual curation is required to prune stale observations.
- **No structured format**: Entries are free-form markdown. A future version could use structured YAML or JSON entries for programmatic filtering (e.g., "show only AAPL observations to the company analysis agent").
