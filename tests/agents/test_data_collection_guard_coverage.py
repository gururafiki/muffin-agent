"""Every data-collecting agent must carry the data-collection guard.

The failure mode is silent by construction: an unguarded collector that
single-shots its schema still returns a well-formed, confident answer. Nothing
errors, nothing looks wrong in the UI, and the numbers are invented. The only
signal is that no tool ran — which is exactly what the guard checks.

So this test asserts the *wiring*, not the behaviour: a new persona added by
copying an existing one, or an analyst refactored later, cannot quietly ship
without the backstop. Behaviour is covered by
``tests/middlewares/test_data_collection_guard.py``.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parents[2] / "src" / "muffin_agent" / "agents"

# Agents that own tools/subagents and produce a structured answer from them.
_COLLECTOR_DIRS = [
    _SRC / "personas_council" / "personas",
    _SRC / "personas_council" / "specialists",
    _SRC / "trading_decision" / "analysts",
]

# `MuffinAgentBuilder(...)` chains that legitimately have no data tools of their
# own, so the guard would bounce twice and change nothing.
_EXEMPT = {
    # Deterministic ToolNode specialists: no LLM in the fetch path at all, so
    # there is no model to nudge (see the graph-authoring rule, Pattern D).
    "technical_analysis.py",
    "sentiment_analysis.py",
}


def _collector_modules() -> list[Path]:
    found: list[Path] = []
    for directory in _COLLECTOR_DIRS:
        for path in sorted(directory.glob("*.py")):
            if path.name.startswith("_") or path.name in _EXEMPT:
                continue
            text = path.read_text()
            # A collector = builds an agent AND binds a structured response.
            if "MuffinAgentBuilder(" in text and ".with_response_format(" in text:
                found.append(path)
    return found


@pytest.mark.unit
def test_the_collector_set_is_not_empty():
    """Guard against the discovery itself silently matching nothing."""
    modules = _collector_modules()
    assert len(modules) >= 20, f"only found {len(modules)} collectors — did paths move?"


@pytest.mark.unit
@pytest.mark.parametrize("path", _collector_modules(), ids=lambda p: p.name)
def test_collector_has_the_data_collection_guard(path: Path):
    """A collector that can single-shot its schema must be guarded."""
    assert ".with_data_collection_guard(" in path.read_text(), (
        f"{path.name} builds an agent with a response format but no "
        ".with_data_collection_guard() — a weak model can single-shot that "
        "schema with zero tool calls and fabricate the figures, and nothing "
        "downstream would notice."
    )


@pytest.mark.unit
@pytest.mark.parametrize("path", _collector_modules(), ids=lambda p: p.name)
def test_guard_is_wired_before_the_model_call_limit(path: Path):
    """Ordering sanity: the guard is part of the builder chain, not stranded.

    Not a correctness requirement (middleware order is resolved by
    ``_assemble_middleware``, not by chain order) — this catches a paste that
    lands the call outside the builder expression entirely.
    """
    text = path.read_text()
    chain = re.search(
        r"MuffinAgentBuilder\(.*?\.with_data_collection_guard\(", text, re.S
    )
    assert chain, (
        f"{path.name}: .with_data_collection_guard() is not in the builder chain"
    )
