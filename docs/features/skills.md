# Skills — Progressive Prompt Disclosure

## Problem

The market regime prompt (`market_regime.jinja`) was approximately 320 lines loaded entirely into the system prompt. This included detailed rubrics for yield curve interpretation, Fama-French factor classification, and regime dimension scoring anchors — sections that are only relevant when the agent has specific data in hand.

Loading the full prompt on every invocation wastes context window tokens and dilutes the agent's attention, particularly for query-only analyses where yield curve or factor data may not be collected at all.

## Solution

Split the prompt into a **core workflow** (always loaded as the system prompt) and **on-demand skills** (loaded by the agent when it needs detailed rubrics). Uses the Deep Agents `skills=` parameter, which enables `SkillsMiddleware` for progressive disclosure.

## How It Works

### Architecture

```
market_regime.jinja (system prompt, ~150 lines)
    │
    ├── Role, subagent table, data collection steps
    ├── High-level Step 4: "read the following skills for detailed rubrics"
    ├── Step 5 reflection checklist
    └── Output schema field descriptions

skills/investment/market-regime/ (loaded on demand)
    │
    ├── yield-curve-analysis/SKILL.md
    │   └── Shape/slope classification tables
    │   └── Credit spread interpretation
    │   └── Real yield and policy rate signals
    │
    ├── factor-regime/SKILL.md
    │   └── Fama-French Z-score thresholds
    │   └── Factor tilt logic (value, quality, momentum, size)
    │   └── Cross-factor consistency checks
    │
    └── regime-synthesis/SKILL.md
        └── 4-dimension scoring anchors (growth, inflation, monetary, liquidity)
        └── Regime label construction examples
        └── Positioning guidance (beta ranges, exposure caps)
```

### CompositeBackend

Skills files live on the local filesystem, but the deep agent's default backend is `OpenSandboxBackend` (for code execution). A `CompositeBackend` routes file reads:

```python
def _composite_backend(runtime):
    sandbox = get_backend(runtime)
    skills_fs = FilesystemBackend(root_dir=_SKILLS_ROOT, virtual_mode=True)
    return CompositeBackend(
        default=sandbox,               # code execution → sandbox
        routes={"/skills/": skills_fs}, # skill files → local filesystem
    )
```

`_SKILLS_ROOT` points to `src/muffin_agent/skills/`.

### SKILL.md format

Each skill follows the Deep Agents SKILL.md specification with YAML frontmatter:

```markdown
---
name: yield-curve-analysis
description: >
  Interpret yield curve shape, slope, credit spreads, real yields,
  and policy rate distance for monetary policy and growth signals.
---

# Yield Curve Analysis

## Shape Classification
...
```

The `SkillsMiddleware` reads the `name` and `description` from frontmatter and exposes them to the agent as available skills. The agent decides when to read each skill based on the data it has collected.

### Prompt reference

The core prompt directs the agent to load skills at the right moment:

```
**Before scoring**, read the following skills for detailed rubrics:
1. **yield-curve-analysis** — interpretation tables for yield curve shape, slope, credit spreads...
2. **factor-regime** — Z-score thresholds and factor tilt logic...
3. **regime-synthesis** — scoring anchors for all 4 dimensions...
```

### Fallback

A full backup of the original prompt is preserved as `market_regime_full.jinja`. If skills loading underperforms (agent doesn't load skills when needed), revert by removing `skills=` from `create_deep_agent` and switching back to the full prompt.

## Files Changed

| File | Change |
|------|--------|
| `src/muffin_agent/skills/investment/market-regime/yield-curve-analysis/SKILL.md` | New: yield curve interpretation rubric |
| `src/muffin_agent/skills/investment/market-regime/factor-regime/SKILL.md` | New: Fama-French Z-score classification |
| `src/muffin_agent/skills/investment/market-regime/regime-synthesis/SKILL.md` | New: 4-dimension regime scoring and positioning guidance |
| `src/muffin_agent/prompts/investment/market_regime.jinja` | Trimmed from ~320 to ~150 lines; detailed rubrics replaced with skill references |
| `src/muffin_agent/prompts/investment/market_regime_full.jinja` | New: full backup of original prompt |
| `src/muffin_agent/agents/investment/market_regime.py` | `CompositeBackend` factory; `skills=["/skills/investment/market-regime/"]` on `create_deep_agent` |
| `tests/agents/test_market_regime.py` | Updated: verify prompt is shorter than full backup, verify skills directory exists, verify composite backend is callable |

## Configuration

| Parameter | Value |
|-----------|-------|
| `skills=` | `["/skills/investment/market-regime/"]` — path relative to `FilesystemBackend` root |
| `backend=` | `_composite_backend` factory — creates `CompositeBackend` at runtime |
| `_SKILLS_ROOT` | `src/muffin_agent/skills/` — resolved from `market_regime.py` location |

### Adding a new skill

1. Create `src/muffin_agent/skills/<domain>/<skill-name>/SKILL.md` with YAML frontmatter
2. Add the skill directory path to the `skills=` list in the relevant `create_deep_agent` call
3. Reference the skill name in the prompt so the agent knows when to load it

## Verification

```bash
pytest tests/agents/test_market_regime.py -v
```

Tests verify:
- Core prompt renders and is shorter than the full backup
- Skills directory exists with expected subdirectories
- `backend` parameter is a callable (composite backend factory)
- `skills` parameter contains the expected path

Manual verification:
- Run a market regime analysis and check LangFuse trace for skill loading events
- Compare output quality: full prompt vs skills-based (qualitative review)

## Criteria Definition Agent — SkillFilterMiddleware

The criteria definition agent (`agents/criteria_definition.py`) is the second agent to use progressive prompt disclosure, with a significantly larger skills corpus (55 skills vs 3 for market regime). It uses the default `SkillsMiddleware` (via `skills=` parameter) for skill parsing, alongside `SkillFilterMiddleware` — a schema-driven `AgentMiddleware` — that pre-filters skills based on a classification provided as input state.

### How it works

Classification is provided as flat state keys, not computed by the agent:

```python
class TickerClassification(AgentState):
    sector: NotRequired[str]
    sub_sector: NotRequired[str]
    market: NotRequired[str]
    stock_type: NotRequired[str]
```

`SkillFilterMiddleware` is parameterised via `__class_getitem__` with the state schema. Category keys are derived automatically from the extra fields (beyond `AgentState`). Two hooks:

1. **`abefore_agent`** — filters `skills_metadata` (set by `SkillsMiddleware`) to only skills matching the classification
2. **`awrap_model_call`** — injects classification context into the system prompt

The agent sees only matched skills (typically 4-6 out of 55) and reads all of them.

### Skills corpus

- **55 SKILL.md files** under `skills/valuation/` (vs 3 for market regime)
- **Flat tag-based naming**: `{tags}` where tags compose freely (e.g. `banking-developed-value`, `emerging`, `growth`)
- **10 supported sectors**: banking, insurance, software-saas, pharmaceuticals, reits, consumer-staples, consumer-discretionary, industrials, energy, telecommunications
- **4 cross-cutting skills**: `guidelines`, `value`, `growth`, `emerging`

### SKILL.md metadata format

Each SKILL.md has category tags in its YAML frontmatter `metadata` field:

```yaml
# Universal skill — no metadata, always matched
---
name: guidelines
description: >
  Quick-reference summary table of primary valuation metrics.
---

# Cross-cutting — one category
---
name: value
description: >
  Common value stock principles and screening questions.
metadata:
  stock_type: value
---

# Full classification — multiple categories
---
name: banking-developed-value
description: >
  Valuation criteria for developed market value banking stocks.
metadata:
  sector: banking
  market: developed
  stock_type: value
---

# Sub-sector exclusive — replaces parent sector skill
---
name: insurance-life-developed-value
description: >
  Life insurance using Embedded Value methodology.
metadata:
  sector: insurance
  sub_sector: life
  market: developed
  stock_type: value
  scope: exclusive
---
```

### Filtering logic

`_filter_skills()` keeps skills whose **ALL** category values match the classification:

- **Universal skills** (no metadata categories) always match
- A skill with `{sector: banking}` matches if the classification has `sector=banking`
- A skill with `{sector: banking, market: developed}` matches only if **both** match
- Results sorted by specificity (fewest categories first → most specific last)

**Category keys**: Derived automatically from the filter schema's extra fields (beyond `AgentState`). For criteria definition: `{"sector", "sub_sector", "market", "stock_type"}`. Supports multiple skill directories with different or overlapping metadata keys — skills whose metadata keys are absent from the classification are naturally excluded.

**Exclusivity**: Skills with `scope: exclusive` in metadata indicate that broader skills in the same category should be skipped. Enforced by the prompt, not by code.

### Workflow

```
Agent starts with flat classification keys in input state
  │
  ├── SkillsMiddleware (built-in): parses 55 SKILL.md files,
  │   writes all to skills_metadata in state
  │
  ├── SkillFilterMiddleware (abefore_agent): filters skills_metadata
  │   to only those matching classification (e.g. 4-6 out of 55)
  │
  ├── SkillsMiddleware (awrap_model_call): lists only filtered skills
  │   in system prompt
  │
  ├── SkillFilterMiddleware (awrap_model_call): injects classification
  │   context into system prompt
  │
  ├── Agent collects contextualization data from 4 subagents
  │
  └── Agent reads all matched skills via read_file
```

### Integration

```python
from ..middlewares import SkillFilterMiddleware, ToolResultCacheMiddleware
from ..utils.backends import get_skills_backend

return create_deep_agent(
    model=llm,
    system_prompt=prompt,
    subagents=subagents,
    backend=get_skills_backend,
    skills=["/skills/valuation/"],  # Built-in SkillsMiddleware
    store=store,
    middleware=[
        ToolResultCacheMiddleware(),
        SkillFilterMiddleware[TickerClassification](),
    ],
    response_format=AutoStrategy(schema=CriteriaDefinitionOutput),
)
```

`skills=` adds the built-in `SkillsMiddleware` which parses SKILL.md files and writes `skills_metadata` to state. `SkillFilterMiddleware[TickerClassification]()` runs after it, filtering `skills_metadata` in `abefore_agent` so only matched skills appear. `get_skills_backend` (from `utils/backends.py`) routes `/skills/` reads to local filesystem.

### Reusing for other agents

`SkillFilterMiddleware` is generic. To use it with another agent:

1. Define an `AgentState` subclass with `NotRequired[str]` fields for each category key
2. Add `metadata` tags to the agent's SKILL.md files
3. Pass `skills=[...]` to `create_deep_agent` for standard skill parsing
4. Pass `backend=get_skills_backend` from `utils/backends.py`
5. Add `SkillFilterMiddleware[YourSchema]()` to the middleware list

### Verification

```bash
# Middleware unit tests (31 tests) + agent integration tests (46 tests)
pytest tests/middlewares/test_skill_suggestion.py tests/agents/test_criteria_definition.py -v
```

## Limitations & Future Work

- **Two agents use skills**: Market regime (3 skills, standard `SkillsMiddleware`) and criteria definition (55 skills, `SkillFilterMiddleware`). Other investment agents (sector analysis, company analysis, forecasting) still load their full prompts. These can be retrofitted with skills as the pattern proves effective.
- **Exclusivity is prompt-driven**: `scope: exclusive` is annotated but not enforced by code. If LLM-driven exclusivity proves unreliable, add `_apply_exclusivity_overrides()` to `_filter_skills()`.
- **No skill versioning**: Skills are loaded from the filesystem with no versioning. If a skill is updated, all future runs use the new version immediately.
- **Revert path**: If skills loading degrades output quality, switch `market_regime.jinja` back to `market_regime_full.jinja` and remove `skills=` and `backend=` overrides from `create_deep_agent`.
