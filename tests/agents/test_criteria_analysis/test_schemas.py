"""Schema validation tests for the criteria-analysis orchestrator."""

import pytest
from pydantic import ValidationError

from muffin_agent.agents.criteria_analysis.schemas import (
    CriteriaAnalysisSynthesis,
    TickerClassificationOutput,
    ValuationMethodologyOutput,
    WeightedBreakdownEntry,
)
from muffin_agent.agents.criterion_evaluation import (
    CriterionEvaluationOutput,
    SubCriterion,
)


@pytest.mark.unit
class TestTickerClassificationOutput:
    def test_minimal_valid(self):
        out = TickerClassificationOutput(
            ticker="JPM",
            sector="banking",
            market="developed",
            stock_type="value",
            rationale="US-based bank with low P/E and high dividend yield.",
            confidence=0.9,
        )
        assert out.sub_sector is None
        assert out.data_sources == []
        assert out.limitations == []

    def test_market_must_be_literal(self):
        with pytest.raises(ValidationError):
            TickerClassificationOutput(
                ticker="X",
                sector="banking",
                market="frontier",  # invalid
                stock_type="value",
                rationale="r",
                confidence=0.5,
            )

    def test_stock_type_must_be_literal(self):
        with pytest.raises(ValidationError):
            TickerClassificationOutput(
                ticker="X",
                sector="banking",
                market="developed",
                stock_type="blend",  # invalid
                rationale="r",
                confidence=0.5,
            )


@pytest.mark.unit
class TestValuationMethodologyOutput:
    def test_minimal_valid(self):
        out = ValuationMethodologyOutput(
            ticker="AAPL",
            methodology_summary=(
                "DCF + EV/EBITDA peer multiple anchored to services growth."
            ),
            additional_criteria=[],
        )
        assert out.sources == []
        assert out.limitations == []

    def test_additional_criteria_use_valuation_criterion(self):
        out = ValuationMethodologyOutput(
            ticker="AAPL",
            methodology_summary="DCF + multiples.",
            additional_criteria=[
                {
                    "name": "Services Revenue Mix",
                    "target_range": ">25%",
                    "weight": 0.05,
                    "assessment_guidance": "Higher mix supports premium multiple.",
                    "data_requirements": ["equity-fundamentals"],
                }
            ],
        )
        assert out.additional_criteria[0].name == "Services Revenue Mix"


@pytest.mark.unit
class TestCriterionEvaluationOutput:
    def test_minimal_valid(self):
        out = CriterionEvaluationOutput(
            criterion_name="ROE",
            score=0.75,
            confidence=0.8,
            signal="positive",
            sub_criteria=[
                SubCriterion(
                    name="Five-year average",
                    weight=1.0,
                    score=0.75,
                    evidence="ROE averaged 14% 2020-2024.",
                )
            ],
            evidence_summary=["ROE 14% (equity-fundamentals 2020-2024)"],
            reasoning="ROE consistently above the cost of equity.",
            counterargument="Recent quarter shows compression to 11%.",
        )
        assert out.limitations == []
        assert out.data_sources == []

    def test_signal_must_be_literal(self):
        with pytest.raises(ValidationError):
            CriterionEvaluationOutput(
                criterion_name="ROE",
                score=0.75,
                confidence=0.8,
                signal="green",  # invalid
                sub_criteria=[],
                evidence_summary=[],
                reasoning="r",
                counterargument="c",
            )


@pytest.mark.unit
class TestCriteriaAnalysisSynthesis:
    def test_minimal_valid(self):
        out = CriteriaAnalysisSynthesis(
            ticker="JPM",
            composite_score=0.62,
            signal="buy",
            weighted_breakdown=[
                WeightedBreakdownEntry(
                    name="ROE",
                    weight=0.4,
                    score=0.75,
                    contribution=0.30,
                    source="skill",
                ),
            ],
            key_positives=["ROE > 10%"],
            key_negatives=["NIM compressing"],
            confidence=0.7,
            thesis_paragraph="JPM screens cheap on a sector-relative basis.",
        )
        assert out.divergences == []

    def test_signal_must_be_literal(self):
        with pytest.raises(ValidationError):
            CriteriaAnalysisSynthesis(
                ticker="X",
                composite_score=0.5,
                signal="watch",  # invalid
                weighted_breakdown=[],
                key_positives=[],
                key_negatives=[],
                confidence=0.5,
                thesis_paragraph="t",
            )

    def test_breakdown_source_must_be_literal(self):
        with pytest.raises(ValidationError):
            WeightedBreakdownEntry(
                name="ROE",
                weight=0.4,
                score=0.75,
                contribution=0.30,
                source="other",  # invalid
            )
