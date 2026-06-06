"""Batch smoke tests for all 13 personas.

Verifies that every registered persona:
* Has a matching Jinja prompt file
* Computes evidence from a realistic data bundle without raising
* Produces a fallback ``hold`` signal when data is missing / errored
* The fallback signal passes Pydantic schema validation
"""

from __future__ import annotations

from typing import Any

import pytest

from muffin_agent.agents.personas import PERSONA_REGISTRY


def _realistic_bundle() -> dict[str, Any]:
    """A reasonably complete bundle that should exercise every scoring path."""
    return {
        "ticker": "TEST",
        "as_of_date": "2025-12-01",
        "market_cap": 50_000_000_000.0,
        "financial_metrics": [
            {
                "return_on_equity": 0.18,
                "return_on_invested_capital": 0.16,
                "debt_to_equity": 0.4,
                "current_ratio": 2.2,
                "operating_margin": 0.22,
                "gross_margin": 0.45,
                "net_margin": 0.15,
                "beta": 1.1,
                "interest_coverage": 8.0,
                "price_to_earnings_ratio": 18.5,
                "asset_turnover": 1.1,
                "ev_to_ebit": 12.0,
                "free_cash_flow_yield": 0.06,
            }
            for _ in range(8)
        ],
        "line_items": {
            "revenue": [10_000, 11_000, 12_000, 13_500, 15_000, 16_800, 18_500, 20_500],
            "gross_profit": [4_500, 5_000, 5_500, 6_200, 7_000, 7_800, 8_600, 9_500],
            "gross_margin": [0.45, 0.45, 0.46, 0.46, 0.47, 0.46, 0.46, 0.46],
            "operating_income": [
                2_200,
                2_400,
                2_700,
                3_000,
                3_300,
                3_700,
                4_100,
                4_500,
            ],
            "operating_margin": [0.22, 0.22, 0.225, 0.222, 0.22, 0.22, 0.22, 0.22],
            "operating_expense": [
                2_300,
                2_600,
                2_800,
                3_200,
                3_700,
                4_100,
                4_500,
                5_000,
            ],
            "ebit": [2_200, 2_400, 2_700, 3_000, 3_300, 3_700, 4_100, 4_500],
            "ebitda": [2_500, 2_750, 3_100, 3_450, 3_800, 4_250, 4_700, 5_150],
            "net_income": [1_500, 1_700, 1_950, 2_200, 2_450, 2_700, 2_950, 3_200],
            "earnings_per_share": [1.5, 1.7, 1.95, 2.2, 2.45, 2.7, 2.95, 3.2],
            "interest_expense": [200, 220, 240, 260, 280, 300, 320, 340],
            "free_cash_flow": [1_200, 1_400, 1_600, 1_800, 2_000, 2_300, 2_550, 2_800],
            "capital_expenditure": [400, 450, 500, 550, 600, 650, 700, 750],
            "depreciation_and_amortization": [
                300,
                330,
                360,
                400,
                440,
                480,
                520,
                560,
            ],
            "dividends_and_other_cash_distributions": [
                -100,
                -110,
                -120,
                -135,
                -150,
                -170,
                -190,
                -210,
            ],
            "issuance_or_purchase_of_equity_shares": [
                -50,
                -60,
                -70,
                -80,
                -90,
                -100,
                -110,
                -120,
            ],
            "total_assets": [
                15_000,
                16_500,
                18_000,
                19_800,
                21_700,
                23_700,
                25_900,
                28_200,
            ],
            "total_liabilities": [
                7_000,
                7_400,
                7_900,
                8_400,
                9_000,
                9_600,
                10_300,
                11_000,
            ],
            "current_assets": [5_000, 5_500, 6_000, 6_600, 7_300, 8_000, 8_800, 9_700],
            "current_liabilities": [
                2_300,
                2_500,
                2_700,
                2_950,
                3_200,
                3_500,
                3_800,
                4_200,
            ],
            "shareholders_equity": [
                8_000,
                9_100,
                10_100,
                11_400,
                12_700,
                14_100,
                15_600,
                17_200,
            ],
            "book_value_per_share": [8.0, 9.1, 10.1, 11.4, 12.7, 14.1, 15.6, 17.2],
            "cash_and_equivalents": [
                3_000,
                3_300,
                3_600,
                4_000,
                4_500,
                5_000,
                5_500,
                6_000,
            ],
            "total_debt": [
                4_000,
                4_200,
                4_500,
                4_800,
                5_200,
                5_600,
                6_000,
                6_500,
            ],
            "outstanding_shares": [1_000] * 8,
            "research_and_development": [
                800,
                900,
                1_000,
                1_100,
                1_250,
                1_400,
                1_550,
                1_700,
            ],
            "goodwill_and_intangible_assets": [
                500,
                520,
                550,
                570,
                600,
                620,
                650,
                680,
            ],
            "return_on_invested_capital": [
                0.15,
                0.15,
                0.16,
                0.16,
                0.17,
                0.17,
                0.17,
                0.18,
            ],
        },
        "market_cap_history": [],
        "insider_trades": [
            {"transaction_shares": 1_000, "insider_name": "Alice"},
            {"transaction_shares": 500, "insider_name": "Bob"},
            {"transaction_shares": -200, "insider_name": "Carol"},
            {"transaction_shares": 1_500, "insider_name": "Dave"},
        ],
        "company_news": [
            {"sentiment": "positive", "title": "Strong quarter"},
            {"sentiment": "positive", "title": "Big contract"},
            {"sentiment": "negative", "title": "Minor headwind"},
            {"sentiment": "neutral", "title": "Routine update"},
            {"sentiment": "positive", "title": "Analyst upgrade"},
        ],
        # 252 daily bars climbing 0.3% / day (synthetic uptrend)
        "prices_1y": [
            {
                "date": f"2025-{(i // 21) + 1:02d}-{(i % 21) + 1:02d}",
                "open": 100 * 1.003**i * 0.999,
                "high": 100 * 1.003**i * 1.005,
                "low": 100 * 1.003**i * 0.995,
                "close": 100 * 1.003**i,
                "volume": 1_000_000 + i * 1000,
            }
            for i in range(252)
        ],
        "benchmark_prices_1y": [],
        "data_quality_notes": [],
    }


@pytest.mark.unit
class TestRegistry:
    def test_registry_contains_all_13(self):
        assert len(PERSONA_REGISTRY) == 13

    def test_every_persona_has_prompt(self):
        """Every persona ships a Jinja prompt file that parses (smoke-render check).

        Refactored to parse-only: v4 persona prompts use persona-specific
        typed Pydantic evidence access (e.g. ``{{ evidence.fundamentals.roe_score }}``),
        so a uniform "render with placeholder vars across all 13 personas"
        smoke test no longer works.  Per-persona render correctness is
        covered in each persona's dedicated test file.
        """
        from jinja2 import Environment, FileSystemLoader

        from muffin_agent.prompts import PROMPTS_DIR

        env = Environment(loader=FileSystemLoader(PROMPTS_DIR))
        for slug in PERSONA_REGISTRY:
            try:
                env.get_template(f"personas/{slug}.jinja")
            except Exception as exc:  # pragma: no cover - failure is the assertion
                pytest.fail(f"Failed to parse personas/{slug}.jinja: {exc}")

    def test_every_persona_has_unique_signal_schema(self):
        schemas = {spec.signal_schema for spec in PERSONA_REGISTRY.values()}
        assert len(schemas) == 13


# Map slug → its internal `_compute_<slug>_facts` function reference.
_FACT_COMPUTERS: dict[str, str] = {
    "warren_buffett": "_compute_buffett_facts",
    "ben_graham": "_compute_graham_facts",
    "cathie_wood": "_compute_wood_facts",
    "charlie_munger": "_compute_munger_facts",
    "bill_ackman": "_compute_ackman_facts",
    "michael_burry": "_compute_burry_facts",
    "mohnish_pabrai": "_compute_pabrai_facts",
    "nassim_taleb": "_compute_taleb_facts",
    "peter_lynch": "_compute_lynch_facts",
    "phil_fisher": "_compute_fisher_facts",
    "rakesh_jhunjhunwala": "_compute_jhunjhunwala_facts",
    "stanley_druckenmiller": "_compute_druckenmiller_facts",
    "aswath_damodaran": "_compute_damodaran_facts",
}


@pytest.mark.unit
@pytest.mark.parametrize("slug", list(_FACT_COMPUTERS))
def test_fact_computer_runs_on_realistic_bundle(slug):
    """Every persona's deterministic fact computer should run without raising."""
    import importlib

    module = importlib.import_module(f"muffin_agent.agents.personas.{slug}")
    func = getattr(module, _FACT_COMPUTERS[slug])
    evidence = func(_realistic_bundle())
    # Sanity: total_score must be non-negative and ≤ max_score
    assert evidence.total_score >= 0
    assert evidence.total_score <= evidence.max_score
    # Pydantic round-trip
    schema = PERSONA_REGISTRY[slug].signal_schema
    dump = evidence.model_dump()
    # Build a fallback signal with this evidence to confirm shape compatibility
    sig = schema(
        agent_id=slug,
        signal="hold",
        confidence=0.5,
        reasoning="test",
        evidence=dump,
    )
    assert sig.signal == "hold"


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("slug", list(_FACT_COMPUTERS))
async def test_node_returns_hold_when_data_bundle_missing(slug):
    """Every persona node should fall back to ``hold`` on missing data — no LLM call."""
    spec = PERSONA_REGISTRY[slug]
    result = await spec.node({"ticker": "TEST"}, {})
    assert "persona_signals" in result
    assert len(result["persona_signals"]) == 1
    sig = result["persona_signals"][0]
    assert sig["agent_id"] == slug
    assert sig["signal"] == "hold"
    assert sig["confidence"] == 0.0
