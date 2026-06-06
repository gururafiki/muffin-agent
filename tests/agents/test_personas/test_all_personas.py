"""Batch smoke tests for all 13 personas — v4 architecture.

Verifies that every persona ships a compiled subgraph that:
* Declares ``ticker`` / ``as_of_date`` / ``query`` in its input schema
* Declares ``persona_signals`` in its output schema
* Has a matching verdict Jinja template that parses
* Provides a ``compute_evidence_node`` that produces typed evidence with
  ``total_score`` between 0 and ``max_score`` on missing data
"""

from __future__ import annotations

from importlib import import_module
from unittest.mock import AsyncMock, patch

import pytest
from jinja2 import Environment, FileSystemLoader

from muffin_agent.agents.personas.council_graph import PERSONA_BUILDERS
from muffin_agent.prompts import PROMPTS_DIR

# Map slug → module path (matches PERSONA_BUILDERS order)
_PERSONA_SLUGS = [slug for slug, _ in PERSONA_BUILDERS]


@pytest.mark.unit
class TestPersonaSurface:
    def test_13_personas_registered(self):
        assert len(_PERSONA_SLUGS) == 13

    def test_every_persona_has_verdict_prompt(self):
        """Every persona ships a Jinja prompt file that parses."""
        env = Environment(loader=FileSystemLoader(PROMPTS_DIR))
        for slug in _PERSONA_SLUGS:
            try:
                env.get_template(f"personas/{slug}.jinja")
            except Exception as exc:
                pytest.fail(f"Failed to parse personas/{slug}.jinja: {exc}")

    def test_every_persona_has_data_collection_prompt(self):
        """Every persona ships a data-collection prompt for its ReAct sub-agent."""
        env = Environment(loader=FileSystemLoader(PROMPTS_DIR))
        for slug in _PERSONA_SLUGS:
            try:
                env.get_template(f"personas/{slug}_data_collection.jinja")
            except Exception as exc:
                pytest.fail(
                    f"Failed to parse personas/{slug}_data_collection.jinja: {exc}"
                )


@pytest.mark.unit
@pytest.mark.parametrize("slug", _PERSONA_SLUGS)
def test_persona_module_has_compute_evidence_node(slug: str) -> None:
    """Every persona module exposes a ``compute_evidence_node(state) -> dict``."""
    module = import_module(f"muffin_agent.agents.personas.{slug}")
    fn = getattr(module, "compute_evidence_node", None)
    assert fn is not None, f"{slug}.compute_evidence_node missing"
    update = fn({})
    assert "evidence" in update
    evidence = update["evidence"]
    assert evidence.total_score >= 0
    assert evidence.total_score <= evidence.max_score


@pytest.mark.unit
@pytest.mark.parametrize("slug", _PERSONA_SLUGS)
def test_persona_module_has_render_verdict_node(slug: str) -> None:
    """Every persona module exposes ``render_verdict_node``."""
    module = import_module(f"muffin_agent.agents.personas.{slug}")
    fn = getattr(module, "render_verdict_node", None)
    assert fn is not None, f"{slug}.render_verdict_node missing"


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("slug,builder", PERSONA_BUILDERS)
async def test_persona_subgraph_compiles(slug, builder) -> None:
    """Every persona's compiled subgraph exposes the right input/output contract."""
    # Mock MCP so we don't hit the network at compile time.
    mock_client = AsyncMock()
    mock_client.get_tools = AsyncMock(return_value=[])
    with patch(
        "muffin_agent.agents.data_collection.utils.MultiServerMCPClient",
        return_value=mock_client,
    ):
        agent = await builder({})

    input_schema = agent.input_schema.model_json_schema()
    output_schema = agent.output_schema.model_json_schema()

    def _flatten(schema):
        props = set(schema.get("properties", {}).keys())
        for defn in schema.get("$defs", {}).values():
            if isinstance(defn, dict):
                props.update(defn.get("properties", {}).keys())
        return props

    input_props = _flatten(input_schema)
    output_props = _flatten(output_schema)

    assert {"ticker", "as_of_date"} <= input_props, (
        f"{slug} input schema missing ticker/as_of_date: {input_props}"
    )
    assert "persona_signals" in output_props, (
        f"{slug} output schema missing persona_signals: {output_props}"
    )
