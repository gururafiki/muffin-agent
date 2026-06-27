"""Enforcing meta-test — every deployable graph must have integration coverage.

Makes "add an E2E integration test" the default for new graphs: any graph
registered in ``langgraph.json`` that is neither covered by a test nor listed in
``PENDING_INTEGRATION_COVERAGE`` fails this suite. New graphs therefore *must*
either ship a ``tests/integration/test_<id>.py`` (see
``docs/integration-testing.md``) or take an explicit, tracked deferral.

The two worked examples (``test_equity_price_collector`` / ``test_persona_peter_lynch``)
cover building-block *subgraphs*, not the deployable graphs in ``langgraph.json`` —
they exist to prove the harness and template the pattern. The deployable graphs
start in ``PENDING_INTEGRATION_COVERAGE`` (backfill tracked in ``roadmap.md``);
move an id into ``COVERED_GRAPHS`` once its test lands.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_LANGGRAPH_JSON = _REPO_ROOT / "langgraph.json"
_INTEGRATION_DIR = Path(__file__).resolve().parent


# graph id (in langgraph.json) → the integration test module that covers it.
COVERED_GRAPHS: dict[str, str] = {
    "council": "test_council_graph_e2e.py",
    "trading_decision": "test_trading_decision.py",
}

# Deployable graphs awaiting integration coverage. Tracked in roadmap.md.
PENDING_INTEGRATION_COVERAGE: set[str] = {
    "stock_evaluation",
    "criteria_analysis",
    "research",
}


def _registered_graph_ids() -> set[str]:
    data = json.loads(_LANGGRAPH_JSON.read_text())
    return set(data.get("graphs", {}).keys())


def test_langgraph_json_is_parseable():
    assert _registered_graph_ids(), "langgraph.json declares no graphs"


def test_every_registered_graph_is_covered_or_pending():
    registered = _registered_graph_ids()
    accounted = set(COVERED_GRAPHS) | PENDING_INTEGRATION_COVERAGE
    missing = registered - accounted
    assert not missing, (
        f"Deployable graph(s) {sorted(missing)} are registered in langgraph.json "
        "but have no integration test. Add tests/integration/test_<id>.py (see "
        "docs/integration-testing.md) and list it in COVERED_GRAPHS, or — if "
        "deferring — add the id to PENDING_INTEGRATION_COVERAGE with a roadmap item."
    )


def test_covered_graph_modules_exist():
    for graph_id, module in COVERED_GRAPHS.items():
        assert (_INTEGRATION_DIR / module).exists(), (
            f"COVERED_GRAPHS[{graph_id!r}] points at missing module {module!r}"
        )


def test_no_stale_pending_entries():
    stale = PENDING_INTEGRATION_COVERAGE - _registered_graph_ids()
    assert not stale, (
        f"PENDING_INTEGRATION_COVERAGE lists {sorted(stale)} which are no longer "
        "in langgraph.json — remove them."
    )


def test_covered_and_pending_are_disjoint():
    overlap = set(COVERED_GRAPHS) & PENDING_INTEGRATION_COVERAGE
    assert not overlap, f"Graphs both covered and pending: {sorted(overlap)}"


@pytest.mark.parametrize("graph_id", sorted(COVERED_GRAPHS))
def test_covered_graphs_are_registered(graph_id: str):
    assert graph_id in _registered_graph_ids()
