# Semantic Validators

## Problem

Investment agents use `AutoStrategy(schema=...)` to enforce structured output via LLM tool-calling. This validates **types** (correct fields, correct types) but not **semantics**. For example:

- Scenario probabilities (base + bull + bear) could sum to 1.5
- Year projections could be unsorted or contain duplicates
- A `company_signal` of "pass" could appear alongside a `quality_signal` of "distressed"
- High confidence could be reported despite many listed limitations

These semantic issues slip through Pydantic type validation and can degrade downstream analysis quality.

## Solution

Post-processing semantic validators that run **after** `structured.model_dump()` but **before** the result is written to graph state. Validators produce warning strings — they never modify the data or raise exceptions.

### Why not Pydantic `@model_validator`?

We intentionally keep validators **outside** the Pydantic models for three reasons:

1. **LLM schema pollution**: `AutoStrategy` serialises the Pydantic model into a JSON schema that the LLM sees as its tool-call contract. Pydantic `@model_validator` decorators add side-effects (raising `ValidationError`, mutating fields) that are invisible in the JSON schema — the LLM cannot see or reason about them. When validation fails, the LLM receives a cryptic tool-call error instead of structured feedback, leading to retry loops or malformed fallback output.

2. **Warning vs rejection**: Pydantic validators are binary — they either pass or raise `ValidationError`. Our semantic checks are **advisory warnings** (e.g., "confidence seems high given limitations"). We want to flag issues without rejecting otherwise usable output. A probability sum of 1.03 is worth noting but should not throw away an entire analysis run.

3. **Separation of concerns**: The Pydantic schemas define the **data contract** between agents (what fields exist, what types they have). Semantic validation is a **quality assurance** concern that belongs in the pipeline layer, not the data model. This keeps schemas clean, reusable, and easy for the LLM to understand.

## How It Works

### Registry pattern

Validators are registered by output schema class name via a `_register()` decorator:

```python
_VALIDATORS: dict[str, type] = {}

def _register(cls_name: str):
    """Register a validator function for *cls_name*."""
    def decorator(fn):
        _VALIDATORS[cls_name] = fn
        return fn
    return decorator

def get_validator(schema_cls: type):
    """Return the validator function for *schema_cls*, or None."""
    return _VALIDATORS.get(schema_cls.__name__)
```

### Integration in `run_deep_agent_node`

After extracting structured output:

```python
result_dict = structured.model_dump()
validator = get_validator(type(structured))
if validator:
    validation_warnings = validator(result_dict)
    if validation_warnings:
        result_dict["_validation_warnings"] = validation_warnings
        logger.warning("Validation warnings for '%s': %s", state_key, validation_warnings)
return {state_key: result_dict}
```

Warnings are attached as a `_validation_warnings` key in the output dict. Downstream consumers can inspect this field for quality flags.

### Validation rules

| Schema | Rule | Warning |
|--------|------|---------|
| `ForecastOutput` | `base + bull + bear` probabilities sum to ~1.0 (tolerance: +/-0.05) | `"Scenario probabilities sum to X.XX, expected ~1.0"` |
| `ForecastOutput` | Projections in each scenario sorted by `year` ascending | `"{scenario} projections are not sorted by year ascending"` |
| `CompanyAnalysisOutput` | `financial_history.years` sorted ascending, no duplicates | `"financial_history.years are not sorted ascending"` or `"...contain duplicates"` |
| `CompanyAnalysisOutput` | `company_signal` consistent with `financial_quality.quality_signal` | `"company_signal='pass' is unusual with financial_quality.quality_signal='distressed'"` |
| All schemas | `confidence > 0.8` with `len(limitations) >= 3` | `"confidence=X.XX seems high given N limitations listed"` |

### Signal consistency map

The `company_signal` to `quality_signal` consistency check uses:

```python
_SIGNAL_QUALITY_MAP = {
    "pass":  {"high", "adequate"},
    "watch": {"adequate", "low"},
    "fail":  {"low", "distressed"},
}
```

## Files Changed

| File | Change |
|------|--------|
| `src/muffin_agent/agents/investment/validators.py` | New: registry, `validate_forecast_output`, `validate_company_analysis_output`, `_check_confidence_vs_limitations` |
| `src/muffin_agent/agents/investment/utils.py` | Call `get_validator()` after `model_dump()`, attach `_validation_warnings` |
| `tests/agents/test_validators.py` | 17 unit tests covering all rules |

## Configuration

No configuration required. Validators run automatically for any schema that has a registered validator function.

### Adding a new validator

To add validation for a new output schema (e.g., `RiskAssessmentOutput`):

```python
@_register("RiskAssessmentOutput")
def validate_risk_assessment_output(data: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    # Add semantic checks here
    warnings.extend(_check_confidence_vs_limitations(data))
    return warnings
```

## Verification

```bash
pytest tests/agents/test_validators.py -v
```

Tests construct dicts with intentionally invalid semantics and verify the correct warnings are produced. Also tests the shared `_check_confidence_vs_limitations` helper and the registry dispatch.

## Limitations & Future Work

- **Warning-only**: Validators never block or modify output. A future improvement could add a "strict mode" that rejects outputs with critical semantic violations (e.g., probability sum > 1.5).
- **No `MarketRegimeOutput` or `SectorViewOutput` validators yet**: These schemas have fewer cross-field consistency constraints, but validators could be added for regime-dimension score ranges or Porter's Five Forces score consistency.
- **No feedback loop**: Warnings are logged and attached to the output dict, but there is no mechanism to feed them back to the LLM for self-correction. A future middleware could intercept validation warnings and prompt the agent to revise.
