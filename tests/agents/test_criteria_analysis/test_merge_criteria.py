"""Tests for the deterministic criteria merge step."""

import pytest

from muffin_agent.agents.criteria_analysis.merge_criteria import (
    _canonical_name,
    merge_criteria_lists,
)


def _criterion(name: str, weight: float = 0.2, **extra) -> dict:
    """Build a minimal valid criterion dict."""
    return {
        "name": name,
        "target_range": "0.8-2.0x",
        "weight": weight,
        "assessment_guidance": "Test guidance",
        "data_requirements": ["equity-fundamentals"],
        **extra,
    }


@pytest.mark.unit
class TestCanonicalName:
    def test_lowercases(self):
        assert _canonical_name("ROE") == "roe"

    def test_strips_punctuation(self):
        assert _canonical_name("P/E Ratio") == "p_e_ratio"

    def test_collapses_multiple_punctuation(self):
        assert _canonical_name("P / E :: Ratio!!!") == "p_e_ratio"

    def test_strips_leading_trailing_underscores(self):
        assert _canonical_name("  hello  world  ") == "hello_world"

    def test_empty_string_returns_empty(self):
        assert _canonical_name("") == ""


@pytest.mark.unit
class TestMergeCriteriaLists:
    def test_disjoint_lists_concatenate(self):
        skill = [_criterion("ROE", 0.5)]
        web = [_criterion("Customer Concentration", 0.5)]
        merged = merge_criteria_lists(skill, web)
        assert {c["name"] for c in merged} == {"ROE", "Customer Concentration"}

    def test_skill_wins_on_duplicate_name(self):
        """When both lists carry the same canonical name the skill version is kept."""
        skill = [_criterion("P/E Ratio", weight=0.3, assessment_guidance="from skill")]
        web = [_criterion("P/E ratio", weight=0.7, assessment_guidance="from web")]
        merged = merge_criteria_lists(skill, web)
        assert len(merged) == 1
        assert merged[0]["assessment_guidance"] == "from skill"
        assert merged[0]["source"] == "skill"

    def test_canonicalisation_handles_punctuation_variants(self):
        """``P/E``, ``P / E``, and ``P E`` are all considered the same criterion."""
        skill = [_criterion("P/E", 0.4)]
        web = [
            _criterion("P / E", 0.3),
            _criterion("p e", 0.3),
        ]
        merged = merge_criteria_lists(skill, web)
        assert len(merged) == 1
        assert merged[0]["name"] == "P/E"

    def test_source_tags_are_set(self):
        skill = [_criterion("ROE")]
        web = [_criterion("Customer Concentration")]
        merged = merge_criteria_lists(skill, web)
        sources = {c["name"]: c["source"] for c in merged}
        assert sources == {"ROE": "skill", "Customer Concentration": "web"}

    def test_weights_renormalise_to_one(self):
        skill = [_criterion("A", 0.4), _criterion("B", 0.4)]  # sums to 0.8
        web = [_criterion("C", 0.4)]
        merged = merge_criteria_lists(skill, web)
        total = sum(c["weight"] for c in merged)
        assert total == pytest.approx(1.0, abs=1e-9)

    def test_weights_renormalise_when_sum_already_one(self):
        skill = [_criterion("A", 0.5)]
        web = [_criterion("B", 0.5)]
        merged = merge_criteria_lists(skill, web)
        assert sum(c["weight"] for c in merged) == pytest.approx(1.0, abs=1e-9)

    def test_zero_weights_get_equal_distribution(self):
        skill = [_criterion("A", 0.0), _criterion("B", 0.0)]
        web = [_criterion("C", 0.0)]
        merged = merge_criteria_lists(skill, web)
        weights = sorted(c["weight"] for c in merged)
        assert weights == pytest.approx([1 / 3, 1 / 3, 1 / 3], abs=1e-9)

    def test_empty_inputs_return_empty(self):
        assert merge_criteria_lists([], []) == []

    def test_only_skill_input(self):
        skill = [_criterion("ROE", 0.6), _criterion("P/B", 0.4)]
        merged = merge_criteria_lists(skill, [])
        assert len(merged) == 2
        assert all(c["source"] == "skill" for c in merged)
        assert sum(c["weight"] for c in merged) == pytest.approx(1.0, abs=1e-9)

    def test_only_web_input(self):
        web = [_criterion("Activist Pressure", 0.5)]
        merged = merge_criteria_lists([], web)
        assert len(merged) == 1
        assert merged[0]["source"] == "web"
        assert merged[0]["weight"] == pytest.approx(1.0, abs=1e-9)

    def test_none_inputs_treated_as_empty(self):
        assert merge_criteria_lists(None, None) == []  # type: ignore[arg-type]

    def test_unnamed_criteria_are_dropped(self):
        skill = [_criterion("ROE", 0.5), _criterion("", 0.5)]
        merged = merge_criteria_lists(skill, [])
        assert len(merged) == 1
        assert merged[0]["name"] == "ROE"
