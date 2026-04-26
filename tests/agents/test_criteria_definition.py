"""Tests for the criteria definition agent."""

from contextlib import ExitStack
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from deepagents import CompiledSubAgent
from pydantic import ValidationError

from muffin_agent.agents.criteria_definition import (
    CriteriaDefinitionOutput,
    TickerClassification,
    ValuationCriterion,
)
from muffin_agent.prompts import render_template
from muffin_agent.utils.backends import _SKILLS_ROOT

# ── Skills directory constants ────────────────────────────────────────────────

_SKILLS_DIR = _SKILLS_ROOT / "valuation"

_EXPECTED_CROSS_CUTTING = {
    "guidelines",
    "value",
    "growth",
    "emerging",
}

_EXPECTED_SECTORS = {
    "banking",
    "insurance",
    "software-saas",
    "pharmaceuticals",
    "reits",
    "consumer-staples",
    "consumer-discretionary",
    "industrials",
    "energy",
    "telecommunications",
}

# ── Minimal valid output dict ─────────────────────────────────────────────────

_VALID_CRITERION = {
    "name": "Price-to-Book Ratio",
    "target_range": "0.8-2.0x",
    "weight": 0.25,
    "assessment_guidance": "Strong: P/B < 1.0 with ROE > 10%. Weak: P/B > 2.0.",
    "data_requirements": ["equity-fundamentals"],
}

_VALID_OUTPUT_DICT = {
    "ticker": "JPM",
    "sector": "Financial Services - Banking",
    "market_type": "developed",
    "stock_type": "value",
    "classification_rationale": (
        "JP Morgan is a US-based bank with P/E below sector median and "
        "high dividend yield, consistent with value classification."
    ),
    "primary_valuation_method": "P/B + P/E Dual-metric",
    "criteria": [
        _VALID_CRITERION,
        {
            "name": "Price-to-Earnings Ratio",
            "target_range": "10-15x",
            "weight": 0.20,
            "assessment_guidance": "Below 10x may signal distress.",
            "data_requirements": ["equity-fundamentals"],
        },
        {
            "name": "Return on Equity",
            "target_range": "10-15%",
            "weight": 0.15,
            "assessment_guidance": "Threshold for cost of equity coverage.",
            "data_requirements": ["equity-fundamentals"],
        },
        {
            "name": "Net Interest Margin",
            "target_range": "2.5-3.5%",
            "weight": 0.15,
            "assessment_guidance": "Core profitability driver for banks.",
            "data_requirements": ["equity-fundamentals"],
        },
        {
            "name": "NPL Ratio",
            "target_range": "<2%",
            "weight": 0.10,
            "assessment_guidance": "Asset quality indicator.",
            "data_requirements": ["equity-fundamentals"],
        },
        {
            "name": "CET1 Ratio",
            "target_range": ">11%",
            "weight": 0.10,
            "assessment_guidance": "Capital adequacy buffer.",
            "data_requirements": ["equity-fundamentals"],
        },
        {
            "name": "Dividend Yield",
            "target_range": "2-4%",
            "weight": 0.05,
            "assessment_guidance": "Income component for value investors.",
            "data_requirements": ["equity-fundamentals"],
        },
    ],
    "screening_questions": [
        "What is the loan book quality trend?",
        "Is capital allocation sustainable?",
    ],
    "valuation_errors_to_avoid": [
        "Applying P/E without adjusting for loan loss provisions",
    ],
    "confidence": 0.85,
    "data_sources": [
        {
            "subagent": "equity-fundamentals",
            "data_retrieved": "P/B, P/E, ROE, NIM",
            "period": "2024-2025",
        },
    ],
    "limitations": [],
}


# ── Prompt template tests ─────────────────────────────────────────────────────


@pytest.mark.unit
class TestPromptTemplate:
    """Test criteria definition prompt template rendering."""

    def test_template_renders(self):
        result = render_template("criteria_definition.jinja")
        assert len(result) > 100

    def test_template_contains_subagents(self):
        result = render_template("criteria_definition.jinja")
        assert "etf-index" in result
        assert "equity-fundamentals" in result
        assert "discovery-screening" in result
        assert "economy-macro" in result
        assert "data-validation" in result

    def test_template_contains_workflow_steps(self):
        result = render_template("criteria_definition.jinja")
        assert "Parse Context" in result
        assert "Collect Contextualization Data" in result
        assert "Validate" in result
        assert "Load Skills" in result
        assert "Reflect" in result

    def test_template_contains_supported_sectors(self):
        result = render_template("criteria_definition.jinja")
        assert "banking" in result
        assert "insurance" in result
        assert "software-saas" in result
        assert "pharmaceuticals" in result
        assert "reits" in result
        assert "consumer-staples" in result
        assert "consumer-discretionary" in result
        assert "industrials" in result
        assert "energy" in result
        assert "telecommunications" in result

    def test_template_instructs_reading_all_skills(self):
        result = render_template("criteria_definition.jinja")
        assert "read_file" in result
        assert "available skills" in result.lower()

    def test_template_contains_grounding_constraint(self):
        result = render_template("criteria_definition.jinja")
        assert "NEVER" in result
        assert "fabricat" in result.lower()

    def test_template_contains_validation_loop(self):
        result = render_template("criteria_definition.jinja")
        assert "data-validation" in result
        assert "proceed" in result
        assert "collect_more_data" in result

    def test_template_contains_reflection_checks(self):
        result = render_template("criteria_definition.jinja")
        assert "Borderline cases" in result
        assert "Multi-sector" in result
        assert "Data recency" in result

    def test_template_instructs_failure_on_unsupported_sector(self):
        result = render_template("criteria_definition.jinja")
        assert "report failure" in result.lower()

    def test_template_does_not_contain_classification_logic(self):
        """Classification is provided as input, not done by the agent."""
        result = render_template("criteria_definition.jinja")
        assert "Value signals" not in result
        assert "Growth signals" not in result
        assert "get_suggested_skills" not in result


# ── ValuationCriterion model tests ────────────────────────────────────────────


@pytest.mark.unit
class TestValuationCriterionModel:
    """Test Pydantic model validation for ValuationCriterion."""

    def test_valid_criterion_validates(self):
        criterion = ValuationCriterion(**_VALID_CRITERION)
        assert criterion.name == "Price-to-Book Ratio"
        assert criterion.weight == 0.25

    def test_missing_required_field_raises(self):
        data = {**_VALID_CRITERION}
        del data["target_range"]
        with pytest.raises(ValidationError):
            ValuationCriterion(**data)


# ── CriteriaDefinitionOutput model tests ──────────────────────────────────────


@pytest.mark.unit
class TestCriteriaDefinitionOutputModel:
    """Test Pydantic model validation for CriteriaDefinitionOutput."""

    def test_valid_full_output_validates(self):
        output = CriteriaDefinitionOutput(**_VALID_OUTPUT_DICT)
        assert output.ticker == "JPM"
        assert output.sector == "Financial Services - Banking"
        assert output.market_type == "developed"
        assert output.stock_type == "value"
        assert output.confidence == 0.85
        assert len(output.criteria) == 7

    def test_invalid_market_type_raises(self):
        data = {**_VALID_OUTPUT_DICT, "market_type": "frontier"}
        with pytest.raises(ValidationError):
            CriteriaDefinitionOutput(**data)

    def test_invalid_stock_type_raises(self):
        data = {**_VALID_OUTPUT_DICT, "stock_type": "blend"}
        with pytest.raises(ValidationError):
            CriteriaDefinitionOutput(**data)

    def test_data_sources_default_to_empty(self):
        data = {**_VALID_OUTPUT_DICT}
        del data["data_sources"]
        output = CriteriaDefinitionOutput(**data)
        assert output.data_sources == []

    def test_limitations_default_to_empty(self):
        data = {**_VALID_OUTPUT_DICT}
        del data["limitations"]
        output = CriteriaDefinitionOutput(**data)
        assert output.limitations == []

    def test_model_dump_serialises_correctly(self):
        output = CriteriaDefinitionOutput(**_VALID_OUTPUT_DICT)
        dumped = output.model_dump()
        assert isinstance(dumped, dict)
        assert dumped["ticker"] == "JPM"
        assert dumped["market_type"] == "developed"
        assert dumped["stock_type"] == "value"
        assert isinstance(dumped["criteria"], list)
        assert len(dumped["criteria"]) == 7
        assert dumped["criteria"][0]["name"] == "Price-to-Book Ratio"

    def test_model_json_schema_contains_all_fields(self):
        schema = CriteriaDefinitionOutput.model_json_schema()
        props = schema["properties"]
        expected_fields = {
            "ticker",
            "sector",
            "market_type",
            "stock_type",
            "classification_rationale",
            "primary_valuation_method",
            "criteria",
            "screening_questions",
            "valuation_errors_to_avoid",
            "confidence",
            "data_sources",
            "limitations",
        }
        assert expected_fields <= set(props.keys())


# ── Skills directory tests ────────────────────────────────────────────────────


@pytest.mark.unit
class TestSkillsDirectory:
    """Test that the skills directory is correctly structured."""

    def test_skills_root_exists(self):
        assert _SKILLS_ROOT.is_dir(), f"Skills root {_SKILLS_ROOT} does not exist"

    def test_criteria_definition_dir_exists(self):
        assert _SKILLS_DIR.is_dir(), f"Skills dir {_SKILLS_DIR} does not exist"

    def test_exactly_55_skill_files(self):
        skill_files = list(_SKILLS_DIR.glob("*/SKILL.md"))
        assert len(skill_files) == 55, (
            f"Expected 55 SKILL.md files, found {len(skill_files)}"
        )

    def test_cross_cutting_skills_exist(self):
        for skill_name in _EXPECTED_CROSS_CUTTING:
            skill_path = _SKILLS_DIR / skill_name / "SKILL.md"
            assert skill_path.is_file(), f"Missing cross-cutting skill: {skill_name}"

    def test_each_sector_has_common_skill(self):
        for sector in _EXPECTED_SECTORS:
            skill_path = _SKILLS_DIR / sector / "SKILL.md"
            assert skill_path.is_file(), f"Missing common skill for sector: {sector}"

    def test_each_sector_has_developed_value_skill(self):
        for sector in _EXPECTED_SECTORS:
            # Insurance has subsectors instead of a single developed-value skill
            if sector == "insurance":
                continue
            skill_path = _SKILLS_DIR / f"{sector}-developed-value" / "SKILL.md"
            assert skill_path.is_file(), (
                f"Missing developed-value skill for sector: {sector}"
            )

    def test_each_sector_has_developed_growth_skill(self):
        for sector in _EXPECTED_SECTORS:
            if sector == "insurance":
                continue
            skill_path = _SKILLS_DIR / f"{sector}-developed-growth" / "SKILL.md"
            assert skill_path.is_file(), (
                f"Missing developed-growth skill for sector: {sector}"
            )

    def test_each_sector_has_emerging_value_skill(self):
        for sector in _EXPECTED_SECTORS:
            skill_path = _SKILLS_DIR / f"{sector}-emerging-value" / "SKILL.md"
            assert skill_path.is_file(), (
                f"Missing emerging-value skill for sector: {sector}"
            )

    def test_each_sector_has_emerging_growth_skill(self):
        for sector in _EXPECTED_SECTORS:
            # Insurance has subsectors; REITs have no EM growth variant
            if sector in ("insurance", "reits"):
                continue
            skill_path = _SKILLS_DIR / f"{sector}-emerging-growth" / "SKILL.md"
            assert skill_path.is_file(), (
                f"Missing emerging-growth skill for sector: {sector}"
            )

    def test_no_valuation_prefix_in_directory_names(self):
        for skill_dir in _SKILLS_DIR.iterdir():
            if skill_dir.is_dir():
                assert not skill_dir.name.startswith("valuation-"), (
                    f"Skill {skill_dir.name} still has 'valuation-' prefix"
                )

    def test_skill_files_have_yaml_frontmatter(self):
        """Every SKILL.md must start with --- frontmatter containing name."""
        for skill_file in _SKILLS_DIR.glob("*/SKILL.md"):
            content = skill_file.read_text()
            assert content.startswith("---"), f"{skill_file} missing YAML frontmatter"
            assert "name:" in content, f"{skill_file} missing 'name' in frontmatter"
            assert "description:" in content, (
                f"{skill_file} missing 'description' in frontmatter"
            )

    def test_insurance_subsector_skills_exist(self):
        """Insurance has subsector-level skills (life vs general)."""
        expected = [
            "insurance-general-developed-value",
            "insurance-general-developed-growth",
            "insurance-life-developed-value",
        ]
        for skill_name in expected:
            skill_path = _SKILLS_DIR / skill_name / "SKILL.md"
            assert skill_path.is_file(), (
                f"Missing insurance subsector skill: {skill_name}"
            )

    def test_industrials_emerging_common_skill_exists(self):
        """Industrials has a common EM skill in addition to value/growth."""
        skill_path = _SKILLS_DIR / "industrials-emerging" / "SKILL.md"
        assert skill_path.is_file()

    def test_telecommunications_emerging_common_skill_exists(self):
        """Telecommunications has a common EM skill."""
        skill_path = _SKILLS_DIR / "telecommunications-emerging" / "SKILL.md"
        assert skill_path.is_file()

    def test_tagged_skills_have_metadata(self):
        """All non-universal skills should have metadata with category tags."""
        import yaml

        for skill_file in _SKILLS_DIR.glob("*/SKILL.md"):
            content = skill_file.read_text()
            parts = content.split("---", 2)
            if len(parts) < 3:
                continue
            frontmatter = yaml.safe_load(parts[1])
            name = frontmatter.get("name", "")
            if name == "guidelines":
                # Universal — no metadata expected
                assert "metadata" not in frontmatter or not frontmatter["metadata"]
                continue
            assert "metadata" in frontmatter, f"{name} missing metadata in frontmatter"
            meta = frontmatter["metadata"]
            assert isinstance(meta, dict), f"{name} metadata is not a dict"
            # Must have at least one category key
            category_keys = {"sector", "sub_sector", "market", "country", "stock_type"}
            assert category_keys & set(meta.keys()), (
                f"{name} has no category keys in metadata: {meta}"
            )

    def test_insurance_subsector_skills_marked_exclusive(self):
        """Insurance sub-sector skills should have scope: exclusive."""
        import yaml

        for name in (
            "insurance-general-developed-value",
            "insurance-general-developed-growth",
            "insurance-life-developed-value",
        ):
            skill_file = _SKILLS_DIR / name / "SKILL.md"
            content = skill_file.read_text()
            parts = content.split("---", 2)
            frontmatter = yaml.safe_load(parts[1])
            meta = frontmatter.get("metadata", {})
            assert meta.get("scope") == "exclusive", (
                f"{name} should have scope: exclusive"
            )
            assert "sub_sector" in meta, f"{name} should have sub_sector"


# ── create_criteria_definition_agent tests ────────────────────────────────────

# Shared patch target prefix
_MOD = "muffin_agent.agents.criteria_definition"


def _mock_subagent_patches():
    """Return a list of patch context managers for all subagent factories."""
    return [
        patch(
            f"{_MOD}.create_etf_index_data_collection_agent",
            new_callable=AsyncMock,
            return_value=MagicMock(),
        ),
        patch(
            f"{_MOD}.create_equity_fundamentals_data_collection_agent",
            new_callable=AsyncMock,
            return_value=MagicMock(),
        ),
        patch(
            f"{_MOD}.create_discovery_screening_data_collection_agent",
            new_callable=AsyncMock,
            return_value=MagicMock(),
        ),
        patch(
            f"{_MOD}.create_economy_macro_data_collection_agent",
            new_callable=AsyncMock,
            return_value=MagicMock(),
        ),
        patch(
            f"{_MOD}.build_validation_subagent",
            new_callable=AsyncMock,
            return_value=CompiledSubAgent(
                name="data-validation",
                description="mock validation",
                runnable=MagicMock(),
            ),
        ),
    ]


@pytest.mark.unit
class TestCreateCriteriaDefinitionAgent:
    """Test agent factory function."""

    @pytest.mark.asyncio
    async def test_creates_agent_with_five_subagents(self):
        mock_etf = MagicMock()
        mock_fundamentals = MagicMock()
        mock_screening = MagicMock()
        mock_macro = MagicMock()
        mock_validation = MagicMock()

        config = MagicMock()
        config.get_llm.return_value = MagicMock()

        with (
            patch(
                f"{_MOD}.ModelConfiguration.from_runnable_config",
                return_value=config,
            ),
            patch(
                f"{_MOD}.create_etf_index_data_collection_agent",
                new_callable=AsyncMock,
                return_value=mock_etf,
            ),
            patch(
                f"{_MOD}.create_equity_fundamentals_data_collection_agent",
                new_callable=AsyncMock,
                return_value=mock_fundamentals,
            ),
            patch(
                f"{_MOD}.create_discovery_screening_data_collection_agent",
                new_callable=AsyncMock,
                return_value=mock_screening,
            ),
            patch(
                f"{_MOD}.create_economy_macro_data_collection_agent",
                new_callable=AsyncMock,
                return_value=mock_macro,
            ),
            patch(
                f"{_MOD}.build_validation_subagent",
                new_callable=AsyncMock,
                return_value=CompiledSubAgent(
                    name="data-validation",
                    description="mock validation",
                    runnable=mock_validation,
                ),
            ),
            patch("muffin_agent.utils.agent_builder.create_deep_agent") as mock_create,
        ):
            mock_create.return_value = MagicMock()

            from muffin_agent.agents.criteria_definition import (
                create_criteria_definition_agent,
            )

            agent = await create_criteria_definition_agent(config)

            mock_create.assert_called_once()
            call_kwargs = mock_create.call_args
            assert call_kwargs.kwargs["model"] == config.get_llm.return_value

            subagents = call_kwargs.kwargs["subagents"]
            assert len(subagents) == 5
            assert subagents[0]["name"] == "etf-index"
            assert subagents[0]["runnable"] is mock_etf
            assert subagents[1]["name"] == "equity-fundamentals"
            assert subagents[1]["runnable"] is mock_fundamentals
            assert subagents[2]["name"] == "discovery-screening"
            assert subagents[2]["runnable"] is mock_screening
            assert subagents[3]["name"] == "economy-macro"
            assert subagents[3]["runnable"] is mock_macro
            assert subagents[4]["name"] == "data-validation"
            assert subagents[4]["runnable"] is mock_validation
            assert agent is mock_create.return_value

    @pytest.mark.asyncio
    async def test_uses_auto_strategy_with_correct_schema(self):
        config = MagicMock()
        config.get_llm.return_value = MagicMock()

        with ExitStack() as stack:
            stack.enter_context(
                patch(
                    f"{_MOD}.ModelConfiguration.from_runnable_config",
                    return_value=config,
                )
            )
            for p in _mock_subagent_patches():
                stack.enter_context(p)
            mock_create = stack.enter_context(
                patch("muffin_agent.utils.agent_builder.create_deep_agent")
            )
            mock_create.return_value = MagicMock()

            from muffin_agent.agents.criteria_definition import (
                create_criteria_definition_agent,
            )

            await create_criteria_definition_agent(config)

            call_kwargs = mock_create.call_args
            response_format = call_kwargs.kwargs["response_format"]
            from langchain.agents.structured_output import AutoStrategy

            assert isinstance(response_format, AutoStrategy)
            assert response_format.schema is CriteriaDefinitionOutput

    @pytest.mark.asyncio
    async def test_passes_skills_param(self):
        """Default SkillsMiddleware is used via skills= parameter."""
        config = MagicMock()
        config.get_llm.return_value = MagicMock()

        with ExitStack() as stack:
            stack.enter_context(
                patch(
                    f"{_MOD}.ModelConfiguration.from_runnable_config",
                    return_value=config,
                )
            )
            for p in _mock_subagent_patches():
                stack.enter_context(p)
            mock_create = stack.enter_context(
                patch("muffin_agent.utils.agent_builder.create_deep_agent")
            )
            mock_create.return_value = MagicMock()

            from muffin_agent.agents.criteria_definition import (
                create_criteria_definition_agent,
            )

            await create_criteria_definition_agent(config)

            call_kwargs = mock_create.call_args
            assert call_kwargs.kwargs["skills"] == ["/skills/valuation/"]

    @pytest.mark.asyncio
    async def test_passes_store_when_provided(self):
        config = MagicMock()
        config.get_llm.return_value = MagicMock()
        mock_store = MagicMock()

        with ExitStack() as stack:
            stack.enter_context(
                patch(
                    f"{_MOD}.ModelConfiguration.from_runnable_config",
                    return_value=config,
                )
            )
            for p in _mock_subagent_patches():
                stack.enter_context(p)
            mock_create = stack.enter_context(
                patch("muffin_agent.utils.agent_builder.create_deep_agent")
            )
            mock_create.return_value = MagicMock()

            from muffin_agent.agents.criteria_definition import (
                create_criteria_definition_agent,
            )

            await create_criteria_definition_agent(config, store=mock_store)

            call_kwargs = mock_create.call_args
            assert call_kwargs.kwargs["store"] is mock_store

    @pytest.mark.asyncio
    async def test_includes_required_middleware(self):
        config = MagicMock()
        config.get_llm.return_value = MagicMock()

        with ExitStack() as stack:
            stack.enter_context(
                patch(
                    f"{_MOD}.ModelConfiguration.from_runnable_config",
                    return_value=config,
                )
            )
            for p in _mock_subagent_patches():
                stack.enter_context(p)
            mock_create = stack.enter_context(
                patch("muffin_agent.utils.agent_builder.create_deep_agent")
            )
            mock_create.return_value = MagicMock()

            from muffin_agent.agents.criteria_definition import (
                create_criteria_definition_agent,
            )

            await create_criteria_definition_agent(config)

            call_kwargs = mock_create.call_args
            middleware = call_kwargs.kwargs["middleware"]
            from langchain.agents.middleware import (
                ModelRetryMiddleware,
                ToolRetryMiddleware,
            )

            from muffin_agent.middlewares import (
                SkillFilterMiddleware,
                ToolErrorHandlerMiddleware,
                ToolResultCacheMiddleware,
            )

            assert len(middleware) == 5
            assert isinstance(middleware[0], ModelRetryMiddleware)
            assert isinstance(middleware[1], ToolErrorHandlerMiddleware)
            assert isinstance(middleware[2], ToolResultCacheMiddleware)
            assert isinstance(middleware[3], ToolRetryMiddleware)
            assert isinstance(middleware[4], SkillFilterMiddleware)
            assert middleware[4].state_schema is TickerClassification


# ── Classification schema tests ──────────────────────────────────────────────


@pytest.mark.unit
class TestClassificationSchemas:
    """Test TickerClassification state schema."""

    def test_ticker_classification_has_expected_extra_fields(self):
        from langchain.agents import AgentState

        base_keys = set(AgentState.__annotations__.keys())
        extra_keys = set(TickerClassification.__annotations__.keys()) - base_keys
        assert extra_keys == {"sector", "sub_sector", "market", "stock_type"}

    def test_ticker_classification_has_agent_state_fields(self):
        """TickerClassification inherits AgentState's fields (messages, etc.)."""
        from langchain.agents import AgentState

        for key in AgentState.__annotations__:
            assert key in TickerClassification.__annotations__
