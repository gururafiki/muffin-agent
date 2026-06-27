"""Per-tool fixture loading — one pluggable file per tool.

Layout (add a new graph's tools by dropping in files; nothing else changes)::

    tests/integration/fixtures/
      openbb/<tool>__<scenario>.json      # OpenBB MCP envelope: {"results": [...], ...}
      firecrawl/<tool>__<scenario>.json   # Firecrawl: a JSON list (list content)

``load_fixture`` returns the value exactly as it reaches the agent as
``ToolMessage.content``:

* **OpenBB** tools return a **JSON string** (the serialized envelope) — matching
  the real ``langchain-mcp-adapters`` text content that the codebase parses with
  ``json.loads(...)["results"]`` (see ``trading_decision/tools.py``).
* **Firecrawl** tools return a **list** (list content) — see the cache
  middleware's strict-content note in CLAUDE.md.

Fixtures are schema-accurate: field names/types are taken from the OpenBB catalogue
(``openbb_mcp_tools.json``, resolved via :func:`openbb_catalogue_path`). Refresh them
against live MCP with ``tests/integration/test_capture_fixtures.py`` (``-m live``).
"""

from __future__ import annotations

import json
from pathlib import Path

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"
_OPENBB_DIR = FIXTURES_DIR / "openbb"
_FIRECRAWL_DIR = FIXTURES_DIR / "firecrawl"

# The OpenBB tool catalogue (``openbb_mcp_tools.json``) is the schema source the
# fixtures + fake-MCP ``inputSchema``s are authored from. It was moved out of this
# repo in the slimming commit (c3705a9) and now ships with the ``openbb-mcp-docker``
# image build — in an umbrella checkout it sits in that sibling submodule. A legacy
# local copy under ``extras/openbb/`` is still honoured. None of these exist in a
# standalone muffin-agent checkout, so callers degrade (fake MCP → permissive schema)
# or skip the catalogue-only tests.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_CATALOGUE_CANDIDATES = (
    _REPO_ROOT.parent / "openbb-mcp-docker" / "openbb_mcp_tools.json",
    _REPO_ROOT / "extras" / "openbb" / "openbb_mcp_tools.json",
)


def openbb_catalogue_path() -> Path | None:
    """First existing OpenBB tool-catalogue path, or ``None`` if none is present."""
    return next((p for p in _CATALOGUE_CANDIDATES if p.is_file()), None)


def _find(tool_name: str, scenario: str) -> tuple[Path, str]:
    """Locate a fixture file, preferring *scenario* then ``generic`` then any.

    Returns ``(path, kind)`` where *kind* is ``"openbb"`` or ``"firecrawl"``.
    """
    for directory, kind in ((_OPENBB_DIR, "openbb"), (_FIRECRAWL_DIR, "firecrawl")):
        for candidate in (
            directory / f"{tool_name}__{scenario}.json",
            directory / f"{tool_name}__generic.json",
        ):
            if candidate.exists():
                return candidate, kind
        matches = sorted(directory.glob(f"{tool_name}__*.json"))
        if matches:
            return matches[0], kind
    raise FileNotFoundError(
        f"No integration fixture for tool {tool_name!r} (scenario {scenario!r}). "
        f"Author one under {FIXTURES_DIR} or capture it via "
        "tests/integration/test_capture_fixtures.py."
    )


def load_fixture(tool_name: str, scenario: str = "aapl") -> str | list:
    """Return the fixture content for *tool_name* as the agent would see it.

    OpenBB → a JSON **string** envelope; Firecrawl → a JSON **list**.
    """
    path, kind = _find(tool_name, scenario)
    data = json.loads(path.read_text())
    if kind == "firecrawl":
        return data  # list content, verbatim
    return json.dumps(data)  # OpenBB string content


def available_tool_names() -> list[str]:
    """All tool names with at least one fixture file (sorted, de-duplicated)."""
    names: set[str] = set()
    for directory in (_OPENBB_DIR, _FIRECRAWL_DIR):
        if directory.exists():
            for p in directory.glob("*__*.json"):
                names.add(p.name.split("__", 1)[0])
    return sorted(names)


def has_fixture(tool_name: str) -> bool:
    """Whether at least one fixture file exists for *tool_name*."""
    try:
        _find(tool_name, "any")
        return True
    except FileNotFoundError:
        return False
