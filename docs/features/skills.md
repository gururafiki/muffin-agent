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

## Limitations & Future Work

- **market_regime only**: Skills are currently implemented only for the market regime agent. Other investment agents (sector analysis, company analysis, forecasting) still load their full prompts. These can be retrofitted with skills as the pattern proves effective.
- **No dynamic skill discovery**: The agent is told which skills exist via the prompt. A more sophisticated approach would let the agent discover skills based on its current data context.
- **No skill versioning**: Skills are loaded from the filesystem with no versioning. If a skill is updated, all future runs use the new version immediately.
- **Revert path**: If skills loading degrades output quality, switch `market_regime.jinja` back to `market_regime_full.jinja` and remove `skills=` and `backend=` overrides from `create_deep_agent`.
