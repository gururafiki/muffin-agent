"""Tests for the prior-findings prompt rendering."""

import pytest
from langchain_core.messages import SystemMessage

from muffin_agent.middlewares.subagent_refinement import (
    CollectionFindings,
    Gap,
    GapReason,
)
from muffin_agent.middlewares.subagent_refinement.prompts import (
    append_block,
    render_prior_findings_block,
)


@pytest.mark.unit
class TestRenderPriorFindingsBlock:
    def test_renders_obtained_and_gaps(self):
        findings = CollectionFindings(
            call_id="abc",
            obtained={"pe": 12.3},
            gaps=[
                Gap(
                    field="ev_ebitda",
                    reason=GapReason.NO_DATA,
                    retry_advice="give_up",
                )
            ],
        )
        block = render_prior_findings_block(findings)
        assert "Prior call context" in block
        assert "call_id=abc" in block
        assert "ev_ebitda" in block
        assert "no_data" in block
        assert "give_up" in block
        # Obtained payload is a JSON code block.
        assert '"pe": 12.3' in block

    def test_no_gaps_section_says_none(self):
        findings = CollectionFindings(call_id="abc", obtained={"pe": 12.3})
        block = render_prior_findings_block(findings)
        assert "(none)" in block


@pytest.mark.unit
class TestAppendBlock:
    def test_no_existing_returns_block_only(self):
        msg = append_block(None, "BLOCK")
        assert isinstance(msg, SystemMessage)
        assert msg.content == "BLOCK"

    def test_existing_text_preserved_and_block_appended(self):
        msg = append_block(SystemMessage(content="base"), "BLOCK")
        assert "base" in msg.content
        assert "BLOCK" in msg.content
        assert msg.content.index("base") < msg.content.index("BLOCK")
