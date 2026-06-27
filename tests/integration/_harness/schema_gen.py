"""Generate schema-correct OpenBB fixture stubs from the tool catalogue.

For the many OpenBB tools whose *realistic values* don't matter to a test — the
LLM-driven collector agents ignore the bytes (the scripted model never reads tool
output), and the result is usually just passed through — a **schema-correct** stub
is enough scaffolding. ``synth_envelope(tool)`` builds one from the real
``outputSchema`` in the OpenBB catalogue (``openbb_mcp_tools.json``, resolved via
``openbb_catalogue_path``) — correct field names + types, lightly humanised values;
``materialize(tool)`` writes it as a fixture.

Tests that *parse* tool values (e.g. the deterministic specialists) should commit
a hand-authored fixture with realistic numbers — it simply overrides the stub.

Regenerate every agent-referenced stub that's missing::

    python -c "import sys; sys.path.insert(0,'tests'); \
        from integration._harness.schema_gen import materialize_missing; \
        print(len(materialize_missing()))"
"""

from __future__ import annotations

import functools
import json
from pathlib import Path
from typing import Any

from .fixtures import FIXTURES_DIR, openbb_catalogue_path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_AGENTS_DIR = _REPO_ROOT / "src" / "muffin_agent" / "agents"

# Provider prefixes seen in OpenBB ``*Data`` model names → lower-case provider id.
_PROVIDER_PREFIXES = (
    "YFinance",
    "FMP",
    "Intrinio",
    "Polygon",
    "Tiingo",
    "Tradier",
    "Benzinga",
    "Cboe",
    "Fred",
    "Finviz",
    "Nasdaq",
    "Tmx",
    "SeekingAlpha",
    "SA",
    "Fmp",
    "AlphaVantage",
    "AV",
    "Biztoc",
    "Tantor",
    "Deribit",
    "EconDB",
    "Imf",
    "IMF",
    "Oecd",
    "OECD",
    "Ecb",
    "ECB",
    "Federal",
    "Stockgrid",
    "Wsj",
    "Cftc",
)


@functools.lru_cache(maxsize=1)
def _catalogue() -> dict[str, dict[str, Any]]:
    path = openbb_catalogue_path()
    if path is None:
        raise FileNotFoundError(
            "OpenBB tool catalogue not found (looked in the openbb-mcp-docker sibling "
            "and extras/openbb/). It was moved out of muffin-agent in c3705a9; "
            "catalogue-dependent tests skip when it is absent — guard callers with "
            "openbb_catalogue_path()."
        )
    data = json.loads(path.read_text())
    tools = data["tools"] if isinstance(data, dict) else data
    return {t["name"]: t for t in tools if isinstance(t, dict)}


def _humanise(field: str, prop: dict[str, Any]) -> Any | None:
    """A friendlier sample value for well-known field names (else ``None``)."""
    low = field.lower()
    is_str = prop.get("type") == "string" or any(
        b.get("type") == "string" for b in prop.get("anyOf", [])
    )
    if not is_str:
        return None
    if low == "symbol" or low.endswith("_symbol"):
        return "AAPL"
    if low == "name":
        return "Apple Inc."
    if "currency" in low:
        return "USD"
    if low == "country":
        return "United States"
    if low == "cik":
        return "0000320193"
    return None


def _sample(prop: dict[str, Any], defs: dict[str, Any], depth: int = 0) -> Any:
    """A schema-correct sample value for one JSON-schema property def."""
    if depth > 4:
        return None
    if "anyOf" in prop:
        non_null = [b for b in prop["anyOf"] if b.get("type") != "null"]
        return _sample(non_null[0], defs, depth) if non_null else None
    if "$ref" in prop:
        return _sample_obj(defs.get(prop["$ref"].split("/")[-1], {}), defs, depth + 1)
    if "enum" in prop and prop["enum"]:
        return prop["enum"][0]
    if prop.get("default") is not None:
        return prop["default"]
    t, fmt = prop.get("type"), prop.get("format")
    if t == "string":
        if fmt == "date":
            return "2026-06-01"
        if fmt == "date-time":
            return "2026-06-01T00:00:00"
        return "sample"
    if t == "integer":
        return 1
    if t == "number":
        return 1.0
    if t == "boolean":
        return False
    if t == "array":
        item = prop.get("items", {})
        return [_sample(item, defs, depth + 1)] if item else []
    if t == "object":
        return {}
    return None


def _sample_obj(
    model: dict[str, Any], defs: dict[str, Any], depth: int = 0
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for field, prop in model.get("properties", {}).items():
        out[field] = _humanise(field, prop)
        if out[field] is None:
            out[field] = _sample(prop, defs, depth)
    return out


def _resolve_results(
    output_schema: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], bool, str]:
    """Resolve the ``results`` provider model, its name, and whether it's a list.

    Handles both OpenBB result shapes: an **array of rows**
    (``anyOf[array{items: anyOf[refs]}, null]``) and a **single object**
    (``anyOf[$ref, null]`` / ``anyOf[anyOf[refs], null]`` — e.g. transcripts,
    SEC maps, options chains). Walks the schema tree to the first ``*Data`` model
    and records whether an ``array`` was crossed on the way.
    """
    defs = output_schema.get("$defs", {})
    queue: list[Any] = [output_schema.get("properties", {}).get("results", {})]
    is_array = False
    while queue:
        node = queue.pop(0)
        if not isinstance(node, dict):
            continue
        if node.get("type") == "array":
            is_array = True
            if "items" in node:
                queue.insert(0, node["items"])
            continue
        if "$ref" in node:
            name = node["$ref"].split("/")[-1]
            if name.endswith("Data"):
                return defs.get(name, {}), defs, is_array, name
            queue.insert(0, defs.get(name, {}))
            continue
        for key in ("anyOf", "oneOf", "allOf"):
            queue.extend(
                b
                for b in node.get(key, [])
                if isinstance(b, dict) and b.get("type") != "null"
            )
    return {}, defs, is_array, ""


def _provider_of(model_name: str) -> str:
    for prefix in _PROVIDER_PREFIXES:
        if model_name.startswith(prefix):
            return prefix.lower()
    return "fmp"


def synth_envelope(tool_name: str) -> dict[str, Any]:
    """Build a schema-correct OpenBB response envelope for *tool_name*.

    ``results`` is a list of rows for array endpoints, or a single object for
    single-result endpoints (transcripts, SEC maps, options chains) — matching
    the tool's real ``outputSchema``.
    """
    tool = _catalogue().get(tool_name)
    if tool is None:
        raise KeyError(f"{tool_name!r} not in the OpenBB catalogue")
    model, defs, is_array, model_name = _resolve_results(tool.get("outputSchema", {}))
    row = _sample_obj(model, defs)
    # A few tools are under-specified in the catalogue (``results: anyOf[{}, null]``
    # — no concrete model). Emit a structurally-valid empty envelope for those.
    if not row:
        results: Any = []
    elif is_array:
        results = [row]
    else:
        results = row
    return {
        "results": results,
        "provider": _provider_of(model_name),
        "warnings": None,
        "chart": None,
        "extra": {},
    }


def materialize(
    tool_name: str, scenario: str = "aapl", *, overwrite: bool = False
) -> Path | None:
    """Write a schema-correct stub fixture; returns the path (None if skipped)."""
    path = FIXTURES_DIR / "openbb" / f"{tool_name}__{scenario}.json"
    if path.exists() and not overwrite:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(synth_envelope(tool_name), indent=2) + "\n")
    return path


def agent_referenced_openbb_tools() -> set[str]:
    """OpenBB catalogue tool names quoted anywhere under ``agents/``."""
    catalogue = set(_catalogue())
    referenced: set[str] = set()
    for py in _AGENTS_DIR.rglob("*.py"):
        text = py.read_text()
        for name in catalogue:
            if f'"{name}"' in text or f"'{name}'" in text:
                referenced.add(name)
    return referenced


def materialize_missing(scenario: str = "aapl") -> list[Path]:
    """Materialise a stub for every agent-referenced OpenBB tool that lacks one."""
    written: list[Path] = []
    for tool_name in sorted(agent_referenced_openbb_tools()):
        path = materialize(tool_name, scenario, overwrite=False)
        if path is not None:
            written.append(path)
    return written
