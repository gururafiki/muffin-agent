"""Tests for SkillFilterMiddleware."""

from typing import NotRequired
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain.agents import AgentState
from langchain_core.messages import SystemMessage

from muffin_agent.middlewares.skill_suggestion.middleware import (
    SkillFilterMiddleware,
)

# ── Helpers ──────────────────────────────────────────────────────────────────

CATEGORY_KEYS = frozenset({"sector", "sub_sector", "market", "stock_type"})


def _skill(name: str, description: str = "", metadata: dict | None = None) -> dict:
    """Build a minimal SkillMetadata-like dict."""
    return {
        "name": name,
        "description": description or f"Description for {name}.",
        "path": f"/skills/valuation/{name}/SKILL.md",
        "metadata": metadata or {},
        "license": None,
        "compatibility": None,
        "allowed_tools": [],
    }


BANKING_DM_VALUE = _skill(
    "banking-developed-value",
    "DM value banking.",
    {"sector": "banking", "market": "developed", "stock_type": "value"},
)
BANKING_COMMON = _skill(
    "banking",
    "Common banking principles.",
    {"sector": "banking"},
)
GUIDELINES = _skill(
    "guidelines",
    "Quick-reference summary table.",
)
VALUE_CROSS = _skill(
    "value",
    "Common value stock principles.",
    {"stock_type": "value"},
)
EMERGING_CROSS = _skill(
    "emerging",
    "Common emerging market adjustments.",
    {"market": "emerging"},
)
PHARMA_EM_GROWTH = _skill(
    "pharmaceuticals-emerging-growth",
    "EM growth pharma.",
    {"sector": "pharmaceuticals", "market": "emerging", "stock_type": "growth"},
)
PHARMA_COMMON = _skill(
    "pharmaceuticals",
    "Common pharma principles.",
    {"sector": "pharmaceuticals"},
)
GROWTH_CROSS = _skill(
    "growth",
    "Common growth stock principles.",
    {"stock_type": "growth"},
)
INSURANCE_LIFE_EXCLUSIVE = _skill(
    "insurance-life-developed-value",
    "Life insurance EV methodology.",
    {
        "sector": "insurance",
        "sub_sector": "life",
        "market": "developed",
        "stock_type": "value",
        "scope": "exclusive",
    },
)
INSURANCE_COMMON = _skill(
    "insurance",
    "Common insurance principles.",
    {"sector": "insurance"},
)

ALL_SKILLS = [
    GUIDELINES,
    VALUE_CROSS,
    GROWTH_CROSS,
    EMERGING_CROSS,
    BANKING_COMMON,
    BANKING_DM_VALUE,
    PHARMA_COMMON,
    PHARMA_EM_GROWTH,
    INSURANCE_COMMON,
    INSURANCE_LIFE_EXCLUSIVE,
]


# ── Test state schemas ───────────────────────────────────────────────────────


class _FullSchema(AgentState):
    sector: NotRequired[str]
    sub_sector: NotRequired[str]
    market: NotRequired[str]
    stock_type: NotRequired[str]


class _MinimalSchema(AgentState):
    sector: NotRequired[str]
    market: NotRequired[str]


class _SectorOnlySchema(AgentState):
    sector: NotRequired[str]


class _EmptySchema(AgentState):
    pass


# ── _filter_skills tests ────────────────────────────────────────────────────


@pytest.mark.unit
class TestFilterSkills:
    """Test the _filter_skills method."""

    def setup_method(self):
        self.mw = SkillFilterMiddleware[_FullSchema]()

    def test_universal_skill_always_matches(self):
        result = self.mw._filter_skills(ALL_SKILLS, {"sector": "banking"})
        names = [s["name"] for s in result]
        assert "guidelines" in names

    def test_single_category_match(self):
        result = self.mw._filter_skills(ALL_SKILLS, {"sector": "banking"})
        names = [s["name"] for s in result]
        assert "banking" in names
        assert "pharmaceuticals" not in names

    def test_multi_category_match(self):
        result = self.mw._filter_skills(
            ALL_SKILLS,
            {"sector": "banking", "market": "developed", "stock_type": "value"},
        )
        names = [s["name"] for s in result]
        assert "banking-developed-value" in names

    def test_partial_classification_matches_broader_skills(self):
        """A classification with only sector should match sector-common skills
        and cross-cutting skills, but not sector+market+type skills."""
        result = self.mw._filter_skills(ALL_SKILLS, {"sector": "banking"})
        names = [s["name"] for s in result]
        assert "banking" in names
        assert "guidelines" in names
        # banking-developed-value requires market and stock_type too
        assert "banking-developed-value" not in names

    def test_cross_cutting_stock_type_matches(self):
        result = self.mw._filter_skills(
            ALL_SKILLS,
            {"sector": "banking", "market": "developed", "stock_type": "value"},
        )
        names = [s["name"] for s in result]
        assert "value" in names
        assert "growth" not in names

    def test_cross_cutting_market_matches(self):
        result = self.mw._filter_skills(
            ALL_SKILLS,
            {"sector": "pharmaceuticals", "market": "emerging", "stock_type": "growth"},
        )
        names = [s["name"] for s in result]
        assert "emerging" in names
        assert "value" not in names

    def test_full_pharma_em_growth_classification(self):
        result = self.mw._filter_skills(
            ALL_SKILLS,
            {"sector": "pharmaceuticals", "market": "emerging", "stock_type": "growth"},
        )
        names = [s["name"] for s in result]
        expected = {
            "guidelines",
            "growth",
            "emerging",
            "pharmaceuticals",
            "pharmaceuticals-emerging-growth",
        }
        assert set(names) == expected

    def test_empty_classification_matches_only_universal(self):
        result = self.mw._filter_skills(ALL_SKILLS, {})
        names = [s["name"] for s in result]
        assert names == ["guidelines"]

    def test_no_match_returns_only_universal(self):
        result = self.mw._filter_skills(ALL_SKILLS, {"sector": "utilities"})
        assert len(result) == 1
        assert result[0]["name"] == "guidelines"

    def test_sorted_by_specificity(self):
        result = self.mw._filter_skills(
            ALL_SKILLS,
            {"sector": "banking", "market": "developed", "stock_type": "value"},
        )
        names = [s["name"] for s in result]
        # Universal (0) → value (1) → banking (1) → banking-dm-value (3)
        assert names.index("guidelines") < names.index(
            "banking-developed-value"
        )
        assert names.index("value") < names.index(
            "banking-developed-value"
        )

    def test_unknown_keys_ignored(self):
        result = self.mw._filter_skills(
            ALL_SKILLS,
            {"sector": "banking", "unknown_key": "foo"},
        )
        names = [s["name"] for s in result]
        assert "banking" in names

    def test_sub_sector_skill_matches(self):
        result = self.mw._filter_skills(
            ALL_SKILLS,
            {
                "sector": "insurance",
                "sub_sector": "life",
                "market": "developed",
                "stock_type": "value",
            },
        )
        names = [s["name"] for s in result]
        assert "insurance-life-developed-value" in names
        assert "insurance" in names  # Also matches (subset)

    def test_sub_sector_mismatch_excludes(self):
        result = self.mw._filter_skills(
            ALL_SKILLS,
            {
                "sector": "insurance",
                "sub_sector": "general",
                "market": "developed",
                "stock_type": "value",
            },
        )
        names = [s["name"] for s in result]
        # life sub_sector doesn't match general
        assert "insurance-life-developed-value" not in names
        assert "insurance" in names


# ── __class_getitem__ / category keys tests ─────────────────────────────────


@pytest.mark.unit
class TestClassGetItem:
    """Test __class_getitem__ parameterisation and category key derivation."""

    def test_full_schema_category_keys(self):
        cls = SkillFilterMiddleware[_FullSchema]
        assert cls._category_keys == frozenset(
            {"sector", "sub_sector", "market", "stock_type"}
        )

    def test_minimal_schema_category_keys(self):
        cls = SkillFilterMiddleware[_MinimalSchema]
        assert cls._category_keys == frozenset({"sector", "market"})

    def test_sector_only_schema_category_keys(self):
        cls = SkillFilterMiddleware[_SectorOnlySchema]
        assert cls._category_keys == frozenset({"sector"})

    def test_empty_schema_has_no_category_keys(self):
        cls = SkillFilterMiddleware[_EmptySchema]
        assert cls._category_keys == frozenset()

    def test_state_schema_is_set(self):
        cls = SkillFilterMiddleware[_FullSchema]
        assert cls.state_schema is _FullSchema

    def test_class_name_includes_schema(self):
        cls = SkillFilterMiddleware[_FullSchema]
        assert "_FullSchema" in cls.__name__


# ── SkillFilterMiddleware instance tests ─────────────────────────────────────


@pytest.mark.unit
class TestSkillFilterMiddleware:
    def test_no_tools_registered(self):
        mw = SkillFilterMiddleware[_FullSchema]()
        assert len(mw.tools) == 0

    def test_category_keys_on_instance(self):
        mw = SkillFilterMiddleware[_FullSchema]()
        assert mw._category_keys == frozenset(
            {"sector", "sub_sector", "market", "stock_type"}
        )

    def test_minimal_schema_instance(self):
        mw = SkillFilterMiddleware[_MinimalSchema]()
        assert mw._category_keys == frozenset({"sector", "market"})


# ── abefore_agent tests ─────────────────────────────────────────────────────


@pytest.mark.unit
class TestAbforeAgent:
    @pytest.mark.asyncio
    async def test_filters_skills_by_classification(self):
        mw = SkillFilterMiddleware[_FullSchema]()
        state = {
            "sector": "banking",
            "market": "developed",
            "stock_type": "value",
            "skills_metadata": ALL_SKILLS,
        }
        result = await mw.abefore_agent(state, MagicMock())
        assert result is not None
        names = [s["name"] for s in result["skills_metadata"]]
        assert "banking-developed-value" in names
        assert "guidelines" in names
        assert "pharmaceuticals" not in names

    @pytest.mark.asyncio
    async def test_returns_none_without_classification(self):
        mw = SkillFilterMiddleware[_FullSchema]()
        state = {"skills_metadata": ALL_SKILLS}
        result = await mw.abefore_agent(state, MagicMock())
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_without_skills(self):
        mw = SkillFilterMiddleware[_FullSchema]()
        state = {"sector": "banking"}
        result = await mw.abefore_agent(state, MagicMock())
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_when_both_missing(self):
        mw = SkillFilterMiddleware[_FullSchema]()
        result = await mw.abefore_agent({}, MagicMock())
        assert result is None

    @pytest.mark.asyncio
    async def test_ignores_unknown_state_keys(self):
        mw = SkillFilterMiddleware[_FullSchema]()
        state = {
            "sector": "banking",
            "bogus": "key",
            "skills_metadata": ALL_SKILLS,
        }
        result = await mw.abefore_agent(state, MagicMock())
        assert result is not None
        names = [s["name"] for s in result["skills_metadata"]]
        assert "banking" in names


# ── awrap_model_call tests ───────────────────────────────────────────────────


@pytest.mark.unit
class TestAwrapModelCall:
    @pytest.mark.asyncio
    async def test_injects_classification_context(self):
        mw = SkillFilterMiddleware[_FullSchema]()

        mock_request = MagicMock()
        mock_request.state = {
            "sector": "banking",
            "market": "developed",
        }
        mock_request.system_message = SystemMessage(content="Base prompt.")

        modified_request = None

        async def handler(request):
            nonlocal modified_request
            modified_request = request
            return MagicMock()

        await mw.awrap_model_call(mock_request, handler)

        mock_request.override.assert_called_once()
        call_kwargs = mock_request.override.call_args.kwargs
        new_sys = call_kwargs["system_message"]
        content_text = "".join(
            b["text"] for b in new_sys.content_blocks if b.get("type") == "text"
        )
        assert "banking" in content_text
        assert "developed" in content_text
        assert "Ticker Classification" in content_text

    @pytest.mark.asyncio
    async def test_passes_through_without_classification(self):
        mw = SkillFilterMiddleware[_FullSchema]()

        mock_request = MagicMock()
        mock_request.state = {}
        expected_response = MagicMock()

        handler = AsyncMock(return_value=expected_response)

        result = await mw.awrap_model_call(mock_request, handler)

        handler.assert_called_once_with(mock_request)
        assert result is expected_response
        mock_request.override.assert_not_called()
