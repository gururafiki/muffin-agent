"""Tests for cross-run memory loading (Commit 6)."""

from unittest.mock import patch

import pytest

from muffin_agent.agents.investment.utils import _AGENTS_MD, load_agent_memory
from muffin_agent.prompts import render_template


class TestLoadAgentMemory:
    """Test ``load_agent_memory()`` behaviour."""

    def test_returns_content_when_file_has_real_entries(self, tmp_path):
        fake_md = tmp_path / "AGENTS.md"
        fake_md.write_text(
            "# Investment Agent Memory\n\n"
            "## Observations\n"
            "2026-03-15 | AAPL | Revenue beat driven by Services segment\n\n"
            "## Sector Trends\n\n"
            "## Model Calibration\n"
        )
        with patch(
            "muffin_agent.agents.investment.utils._AGENTS_MD", fake_md
        ):
            result = load_agent_memory()
        assert "AAPL" in result
        assert "Revenue beat" in result

    def test_returns_empty_for_seed_template(self):
        """Seed-only AGENTS.md should not pollute the prompt."""
        result = load_agent_memory()
        # The actual file on disk is the seed template with no real entries
        assert result == ""

    def test_returns_empty_when_file_missing(self, tmp_path):
        fake_md = tmp_path / "nonexistent.md"
        with patch(
            "muffin_agent.agents.investment.utils._AGENTS_MD", fake_md
        ):
            result = load_agent_memory()
        assert result == ""

    def test_returns_content_with_sector_trends(self, tmp_path):
        fake_md = tmp_path / "AGENTS.md"
        fake_md.write_text(
            "# Investment Agent Memory\n\n"
            "## Observations\n\n"
            "## Sector Trends\n"
            "2026-03-10 | TECH | AI capex cycle accelerating\n\n"
            "## Model Calibration\n"
        )
        with patch(
            "muffin_agent.agents.investment.utils._AGENTS_MD", fake_md
        ):
            result = load_agent_memory()
        assert "AI capex cycle" in result

    def test_returns_content_with_model_calibration(self, tmp_path):
        fake_md = tmp_path / "AGENTS.md"
        fake_md.write_text(
            "# Investment Agent Memory\n\n"
            "## Observations\n\n"
            "## Sector Trends\n\n"
            "## Model Calibration\n"
            "CPI data from FRED often lags by 2 weeks\n"
        )
        with patch(
            "muffin_agent.agents.investment.utils._AGENTS_MD", fake_md
        ):
            result = load_agent_memory()
        assert "CPI data" in result


class TestAgentsMdExists:
    """Verify the seed file ships with the package."""

    def test_agents_md_exists(self):
        assert _AGENTS_MD.exists(), f"AGENTS.md not found at {_AGENTS_MD}"

    def test_agents_md_has_expected_sections(self):
        content = _AGENTS_MD.read_text()
        assert "## Observations" in content
        assert "## Sector Trends" in content
        assert "## Model Calibration" in content


class TestMemoryTemplateIntegration:
    """Verify the ``_memory.jinja`` partial renders correctly."""

    def test_memory_included_when_provided(self):
        prompt = render_template(
            "investment/market_regime.jinja",
            memory="2026-03-15 | AAPL | Strong services revenue",
        )
        assert "Prior Observations" in prompt
        assert "Strong services revenue" in prompt

    def test_memory_omitted_when_empty(self):
        prompt = render_template("investment/market_regime.jinja", memory="")
        assert "Prior Observations" not in prompt

    def test_memory_omitted_when_not_passed(self):
        prompt = render_template("investment/market_regime.jinja")
        assert "Prior Observations" not in prompt

    @pytest.mark.parametrize(
        "template",
        [
            "investment/market_regime.jinja",
            "investment/sector_analysis.jinja",
            "investment/company_analysis.jinja",
            "investment/forecasting.jinja",
        ],
    )
    def test_all_templates_support_memory(self, template):
        """All 4 investment templates should accept and render memory."""
        # forecasting.jinja requires extra kwargs
        kwargs = {"memory": "test observation"}
        if "forecasting" in template:
            kwargs.update(
                base_probability=0.6,
                bull_probability=0.25,
                bear_probability=0.15,
                company_signal="pass",
            )
        prompt = render_template(template, **kwargs)
        assert "test observation" in prompt
