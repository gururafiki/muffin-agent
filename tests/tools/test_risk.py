"""Unit tests for risk tools (beta, VaR/CVaR, Sharpe/Sortino, max drawdown)."""


import pytest

from muffin_agent.tools.risk import (
    compute_beta,
    compute_max_drawdown,
    compute_sharpe_sortino,
    compute_var_cvar,
)

# ── compute_beta ──────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestComputeBeta:
    def test_beta_one_for_identical_series(self):
        r = [0.01, -0.02, 0.03, -0.01, 0.02]
        result = compute_beta.invoke({"returns": r, "market_returns": r})
        assert result["beta"] == pytest.approx(1.0, abs=1e-9)
        assert result["alpha_annualized"] == pytest.approx(0.0, abs=1e-9)
        assert result["r_squared"] == pytest.approx(1.0, abs=1e-9)

    def test_beta_zero_for_uncorrelated_constant_market(self):
        # market has zero variance → beta should be None
        r = [0.01, -0.01, 0.02]
        m = [0.005, 0.005, 0.005]
        result = compute_beta.invoke({"returns": r, "market_returns": m})
        assert result["beta"] is None

    def test_beta_known_value(self):
        # Build a series where stock = 2 * market (beta should be 2.0)
        market = [0.01, -0.02, 0.03, 0.00, -0.01]
        stock = [2 * m for m in market]
        result = compute_beta.invoke({"returns": stock, "market_returns": market})
        assert result["beta"] == pytest.approx(2.0, abs=1e-9)
        assert result["r_squared"] == pytest.approx(1.0, abs=1e-9)

    def test_alpha_annualised_weekly(self):
        # stock = 0.001 + market → alpha_per_period = 0.001,
        # annualised weekly = 0.001 * 52
        market = [0.01, -0.02, 0.03, 0.00, -0.01, 0.02]
        stock = [m + 0.001 for m in market]
        result = compute_beta.invoke(
            {"returns": stock, "market_returns": market, "frequency": "weekly"}
        )
        assert result["beta"] == pytest.approx(1.0, abs=1e-6)
        assert result["alpha_annualized"] == pytest.approx(0.001 * 52, abs=1e-6)

    def test_alpha_annualised_daily(self):
        market = [0.001, -0.002, 0.003, 0.000, -0.001, 0.002]
        stock = [m + 0.0001 for m in market]
        result = compute_beta.invoke(
            {"returns": stock, "market_returns": market, "frequency": "daily"}
        )
        assert result["alpha_annualized"] == pytest.approx(0.0001 * 252, abs=1e-6)

    def test_fewer_than_two_observations(self):
        result = compute_beta.invoke({"returns": [0.01], "market_returns": [0.01]})
        assert result["beta"] is None
        assert result["alpha_annualized"] is None
        assert result["r_squared"] is None

    def test_empty_series(self):
        result = compute_beta.invoke({"returns": [], "market_returns": []})
        assert result["beta"] is None

    def test_mismatched_lengths_uses_shorter(self):
        r = [0.01, -0.02, 0.03, 0.00, -0.01]
        m = [0.01, -0.02]
        result = compute_beta.invoke({"returns": r, "market_returns": m})
        # Only 2 observations used; result should still be valid
        assert result["beta"] is not None

    def test_output_schema_in_extras(self):
        schema = compute_beta.extras["output_schema"]
        assert "properties" in schema
        for key in ("beta", "alpha_annualized", "r_squared"):
            assert key in schema["properties"]


# ── compute_var_cvar ──────────────────────────────────────────────────────────


@pytest.mark.unit
class TestComputeVarCvar:
    def test_known_values_95pct_1month(self):
        # 25% annual vol, $100 price, 95% confidence, 1-month horizon
        # sigma_monthly = 0.25 * sqrt(1/12) ≈ 0.07217
        # z_0.95 ≈ 1.6449
        # VaR_pct ≈ 1.6449 * 7.217 ≈ 11.87%
        result = compute_var_cvar.invoke(
            {
                "annualized_vol_pct": 25.0,
                "current_price": 100.0,
                "confidence": 0.95,
                "horizon_months": 1.0,
            }
        )
        assert result["var_pct"] == pytest.approx(11.87, abs=0.05)
        assert result["var_dollar"] == pytest.approx(11.87, abs=0.05)
        assert result["cvar_pct"] > result["var_pct"]  # CVaR always > VaR

    def test_cvar_greater_than_var(self):
        result = compute_var_cvar.invoke(
            {
                "annualized_vol_pct": 30.0,
                "current_price": 50.0,
            }
        )
        assert result["cvar_pct"] > result["var_pct"]

    def test_dollar_equals_pct_times_price(self):
        result = compute_var_cvar.invoke(
            {
                "annualized_vol_pct": 20.0,
                "current_price": 200.0,
            }
        )
        assert result["var_dollar"] == pytest.approx(
            200.0 * result["var_pct"] / 100, abs=0.01
        )
        assert result["cvar_dollar"] == pytest.approx(
            200.0 * result["cvar_pct"] / 100, abs=0.01
        )

    def test_longer_horizon_higher_var(self):
        base = compute_var_cvar.invoke(
            {
                "annualized_vol_pct": 25.0,
                "current_price": 100.0,
                "horizon_months": 1.0,
            }
        )
        longer = compute_var_cvar.invoke(
            {
                "annualized_vol_pct": 25.0,
                "current_price": 100.0,
                "horizon_months": 3.0,
            }
        )
        assert longer["var_pct"] > base["var_pct"]

    def test_zero_vol_returns_none(self):
        result = compute_var_cvar.invoke(
            {
                "annualized_vol_pct": 0.0,
                "current_price": 100.0,
            }
        )
        assert result["var_pct"] is None

    def test_negative_price_returns_none(self):
        result = compute_var_cvar.invoke(
            {
                "annualized_vol_pct": 20.0,
                "current_price": -10.0,
            }
        )
        assert result["var_pct"] is None

    def test_horizon_scaling(self):
        # VaR should scale by sqrt(horizon): var(4m) ≈ var(1m) * sqrt(4) = 2 * var(1m)
        r1 = compute_var_cvar.invoke(
            {"annualized_vol_pct": 25.0, "current_price": 100.0, "horizon_months": 1.0}
        )
        r4 = compute_var_cvar.invoke(
            {"annualized_vol_pct": 25.0, "current_price": 100.0, "horizon_months": 4.0}
        )
        assert r4["var_pct"] == pytest.approx(r1["var_pct"] * 2.0, abs=0.01)

    def test_output_schema_in_extras(self):
        schema = compute_var_cvar.extras["output_schema"]
        assert "properties" in schema
        for key in ("var_pct", "var_dollar", "cvar_pct", "cvar_dollar"):
            assert key in schema["properties"]


# ── compute_sharpe_sortino ────────────────────────────────────────────────────


@pytest.mark.unit
class TestComputeSharpeSortino:
    # 52 weekly returns at a constant 1% — Sortino = Sharpe (no downside)
    _constant_positive = [0.01] * 52

    def test_all_positive_returns_sortino_none(self):
        """When all excess returns are non-negative, Sortino has no downside → None."""
        result = compute_sharpe_sortino.invoke(
            {
                "returns": self._constant_positive,
                "risk_free_rate_annual": 0.0,
                "frequency": "weekly",
            }
        )
        # All returns are identical and positive → zero variance → Sharpe is None
        assert result["sortino_ratio"] is None

    def test_sharpe_positive_for_positive_mean_excess(self):
        # Returns consistently above zero rf rate
        returns = [0.01, 0.02, -0.005, 0.015, 0.01] * 10
        result = compute_sharpe_sortino.invoke(
            {
                "returns": returns,
                "risk_free_rate_annual": 0.0,
                "frequency": "weekly",
            }
        )
        assert result["sharpe_ratio"] is not None
        assert result["sharpe_ratio"] > 0

    def test_sharpe_negative_for_negative_mean(self):
        returns = [-0.01, -0.02, -0.005, -0.015] * 10
        result = compute_sharpe_sortino.invoke(
            {
                "returns": returns,
                "risk_free_rate_annual": 0.0,
                "frequency": "weekly",
            }
        )
        assert result["sharpe_ratio"] < 0

    def test_sortino_higher_than_sharpe_for_positive_mean_mixed_returns(self):
        # Positive mean excess returns with occasional deep negatives.
        # Sortino denominator (downside deviation) < total std → Sortino > Sharpe.
        returns = [0.05, 0.04, 0.06, -0.02, 0.05, 0.04, 0.06, -0.01] * 5
        result = compute_sharpe_sortino.invoke(
            {
                "returns": returns,
                "risk_free_rate_annual": 0.0,
                "frequency": "weekly",
            }
        )
        if result["sortino_ratio"] is not None and result["sharpe_ratio"] is not None:
            assert result["sortino_ratio"] > result["sharpe_ratio"]

    def test_daily_vs_weekly_frequency(self):
        returns_daily = [0.001, -0.002, 0.003, 0.001, -0.001] * 20
        r_weekly = compute_sharpe_sortino.invoke(
            {
                "returns": returns_daily,
                "risk_free_rate_annual": 0.04,
                "frequency": "weekly",
            }
        )
        r_daily = compute_sharpe_sortino.invoke(
            {
                "returns": returns_daily,
                "risk_free_rate_annual": 0.04,
                "frequency": "daily",
            }
        )
        # daily uses 252 annualisation, weekly uses 52 → different values
        if r_weekly["sharpe_ratio"] and r_daily["sharpe_ratio"]:
            assert r_weekly["sharpe_ratio"] != pytest.approx(
                r_daily["sharpe_ratio"], abs=0.1
            )

    def test_fewer_than_two_observations(self):
        result = compute_sharpe_sortino.invoke(
            {
                "returns": [0.01],
                "risk_free_rate_annual": 0.04,
            }
        )
        assert result["sharpe_ratio"] is None
        assert result["sortino_ratio"] is None

    def test_output_schema_in_extras(self):
        schema = compute_sharpe_sortino.extras["output_schema"]
        assert "properties" in schema
        assert "sharpe_ratio" in schema["properties"]
        assert "sortino_ratio" in schema["properties"]


# ── compute_max_drawdown ──────────────────────────────────────────────────────


@pytest.mark.unit
class TestComputeMaxDrawdown:
    def test_known_drawdown(self):
        # Peak at 110, trough at 80 → drawdown = (80 - 110) / 110 ≈ -0.2727
        prices = [100.0, 110.0, 80.0, 95.0]
        result = compute_max_drawdown.invoke({"prices": prices})
        assert result == pytest.approx((80 - 110) / 110, abs=1e-9)

    def test_monotonically_increasing_zero_drawdown(self):
        prices = [100.0, 105.0, 110.0, 120.0, 130.0]
        result = compute_max_drawdown.invoke({"prices": prices})
        assert result == pytest.approx(0.0, abs=1e-9)

    def test_monotonically_decreasing(self):
        prices = [100.0, 90.0, 80.0, 70.0]
        result = compute_max_drawdown.invoke({"prices": prices})
        # Peak at 100, trough at 70 → (70 - 100) / 100 = -0.30
        assert result == pytest.approx(-0.30, abs=1e-9)

    def test_single_price_returns_none(self):
        assert compute_max_drawdown.invoke({"prices": [100.0]}) is None

    def test_empty_returns_none(self):
        assert compute_max_drawdown.invoke({"prices": []}) is None

    def test_result_is_negative_or_zero(self):
        import random

        random.seed(42)
        prices = [100.0 * (1 + random.uniform(-0.05, 0.05)) for _ in range(50)]
        prices = [abs(p) for p in prices]  # ensure positive
        result = compute_max_drawdown.invoke({"prices": prices})
        assert result is not None
        assert result <= 0.0

    def test_two_prices_declining(self):
        result = compute_max_drawdown.invoke({"prices": [100.0, 80.0]})
        assert result == pytest.approx(-0.20, abs=1e-9)
