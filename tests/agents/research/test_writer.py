"""Tests for the writer stage's pure finalize node.

The writer itself is now a compiled agent added via ``add_node``. What remains worth
testing is ``finalize_output_node``: the model is asked to echo ``task_type`` /
``mode_used`` back into its public output, and a mislabel there is a silent lie to
the caller — so the pipeline overwrites both with its own truth.
"""

from __future__ import annotations

import pytest

from muffin_agent.agents.research.nodes import writer as writer_module
from muffin_agent.agents.research.schemas import ResearchOutput


@pytest.mark.unit
class TestFinalizeOutputNode:
    def test_returns_validated_output(self, sample_research_output: ResearchOutput):
        result = writer_module.finalize_output_node(
            {
                "output": sample_research_output.model_dump(),
                "task_type": "how_to",
                "mode": "balanced",
            }
        )
        output = result["output"]
        # Re-validation must pass.
        ResearchOutput.model_validate(output)
        assert output["task_type"] == "how_to"
        assert output["mode_used"] == "balanced"

    def test_task_type_and_mode_pinned_from_state(
        self, sample_research_output: ResearchOutput
    ):
        """A writer that mislabels its own metadata gets corrected."""
        mislabelled = sample_research_output.model_copy(
            update={"task_type": "wrong", "mode_used": "wrong"}
        ).model_dump()
        result = writer_module.finalize_output_node(
            {"output": mislabelled, "task_type": "comparison", "mode": "quality"}
        )
        assert result["output"]["task_type"] == "comparison"
        assert result["output"]["mode_used"] == "quality"

    def test_no_output_is_a_noop(self):
        """Nothing to pin — never fabricate an output the writer didn't produce."""
        assert writer_module.finalize_output_node({"task_type": "summary"}) == {}

    def test_defaults_when_state_lacks_metadata(
        self, sample_research_output: ResearchOutput
    ):
        result = writer_module.finalize_output_node(
            {"output": sample_research_output.model_dump()}
        )
        assert result["output"]["task_type"] == "research_report"
        assert result["output"]["mode_used"] == "balanced"
