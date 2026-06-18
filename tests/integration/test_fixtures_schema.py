"""Sanity guards for the fixture library + the schema-stub generator.

Cheap insurance: catches a corrupted/committed-bad fixture and a generator that
drifts from the OpenBB catalogue (e.g. after a catalogue refresh).
"""

from __future__ import annotations

import json

import pytest

from ._harness.fixtures import FIXTURES_DIR
from ._harness.schema_gen import (
    _catalogue,
    _resolve_results,
    agent_referenced_openbb_tools,
    synth_envelope,
)


def _openbb_fixture_files() -> list:
    return sorted((FIXTURES_DIR / "openbb").glob("*.json"))


@pytest.mark.parametrize("path", _openbb_fixture_files(), ids=lambda p: p.name)
def test_openbb_fixture_is_a_valid_envelope(path) -> None:
    """Every committed OpenBB fixture is a well-formed response envelope.

    ``results`` is a list (array endpoints) or a dict (single-result endpoints).
    """
    data = json.loads(path.read_text())
    assert isinstance(data, dict), f"{path.name}: not a JSON object"
    assert isinstance(data.get("results"), (list, dict)), (
        f"{path.name}: 'results' is neither a list nor an object"
    )


def test_generator_covers_every_schematizable_tool() -> None:
    """schema_gen fills every agent-used tool that HAS a concrete output model.

    Locks the generator against catalogue drift: any referenced tool whose
    ``outputSchema`` resolves to a provider data model must yield a non-empty
    result. Tools the catalogue under-specifies (``results: anyOf[{}, null]``)
    are legitimately skipped — there's nothing to synthesise.
    """
    catalogue = _catalogue()
    referenced = agent_referenced_openbb_tools()
    assert referenced, "no agent-referenced OpenBB tools found"
    empty: list[str] = []
    for tool in sorted(referenced):
        model, _defs, _is_array, _name = _resolve_results(
            catalogue[tool].get("outputSchema", {})
        )
        if not model:  # catalogue under-specifies this tool — nothing to fill
            continue
        results = synth_envelope(tool).get("results")
        ok = (
            (bool(results) and bool(results[0]))
            if isinstance(results, list)
            else bool(results)
        )
        if not ok:
            empty.append(tool)
    assert not empty, f"generator produced empty rows for schematizable tools: {empty}"
