"""Unit tests for econ package: Production, AdjustmentCosts, CashFlow,
BondPricing, IssuanceCosts, DebtFlow, SteadyState, Collateral."""

from __future__ import annotations

import numpy as np
import tensorflow as tf

tf.config.set_visible_devices([], 'GPU')

import pytest

from econ_models.config.economic_params import EconomicParams
from econ_models.econ import (
    ProductionFunctions,
    AdjustmentCostCalculator,
    CashFlowCalculator,
    BondPricingCalculator,
    IssuanceCostCalculator,
    DebtFlowCalculator,
    SteadyStateCalculator,
    CollateralCalculator,
)


def _make_params(**overrides):
    defaults = dict(
        discount_factor=0.96,
        capital_share=0.3,
        depreciation_rate=0.1,
        productivity_persistence=0.9,
        productivity_std_dev=0.05,
        adjustment_cost_convex=1.0,
        adjustment_cost_fixed=0.02,
        equity_issuance_cost_fixed=0.08,
        equity_issuance_cost_linear=0.1,
        default_cost_proportional=0.3,
        corporate_tax_rate=0.2,
        risk_free_rate=0.04,
        collateral_recovery_fraction=0.5,
    )
    defaults.update(overrides)
    return EconomicParams(**defaults)


# ─────────────── Production ───────────────


class TestCobbDouglas:
    """Tests for ProductionFunctions.cobb_douglas: Y = Z * K^theta."""

    def test_known_value(self):
        params = _make_params(capital_share=0.5)
        k = tf.constant([[4.0]])
        z = tf.constant([[1.0]])
        y = ProductionFunctions.cobb_douglas(k, z, params)
        np.testing.assert_allclose(float(y), 2.0, atol=1e-5)

    def test_higher_z_more_output(self):
        params = _make_params()
        k = tf.constant([[2.0]])
        y_low = ProductionFunctions.cobb_douglas(k, tf.constant([[0.5]]), params)
        y_high = ProductionFunctions.cobb_douglas(k, tf.constant([[1.5]]), params)
        assert float(y_high) > float(y_low)

    def test_batch_shape(self):
        params = _make_params()
        k = tf.random.uniform((8, 1), 0.5, 3.0)
        z = tf.random.uniform((8, 1), 0.5, 1.5)
        y = ProductionFunctions.cobb_douglas(k, z, params)
        assert y.shape == (8, 1)

    def test_output_positive(self):
        params = _make_params()
        k = tf.random.uniform((100, 1), 0.1, 5.0)
        z = tf.random.uniform((100, 1), 0.1, 3.0)
        y = ProductionFunctions.cobb_douglas(k, z, params)
        assert tf.reduce_all(y > 0)


class TestCalculateInvestment:
    """Tests for ProductionFunctions.calculate_investment: I = K' - (1-δ)*K."""

    def test_zero_growth(self):
        """K' = (1-δ)*K → I = 0 (just replacement)."""
        params = _make_params(depreciation_rate=0.1)
        k = tf.constant([[2.0]])
        k_prime = k * (1.0 - params.depreciation_rate)
        invest = ProductionFunctions.calculate_investment(k, k_prime, params)
        np.testing.assert_allclose(float(invest), 0.0, atol=1e-6)

    def test_positive_investment(self):
        params = _make_params(depreciation_rate=0.1)
        k = tf.constant([[2.0]])
        k_prime = tf.constant([[2.5]])
        invest = ProductionFunctions.calculate_investment(k, k_prime, params)
        expected = 2.5 - 0.9 * 2.0  # 2.5 - 1.8 = 0.7
        np.testing.assert_allclose(float(invest), expected, atol=1e-5)


# ─────────────── Adjustment Costs ───────────────


class TestAdjustmentCostCalculator:
    """Tests for AdjustmentCostCalculator.calculate."""

    def test_zero_investment_zero_convex(self):
        params = _make_params(adjustment_cost_convex=1.0, adjustment_cost_fixed=0.0)
        invest = tf.constant([[0.0]])
        k = tf.constant([[2.0]])
        total, marginal = AdjustmentCostCalculator.calculate(invest, k, params)
        np.testing.assert_allclose(float(total), 0.0, atol=1e-6)
        np.testing.assert_allclose(float(marginal), 0.0, atol=1e-6)

    def test_convex_cost_formula(self):
        """ψ0/2 * I²/K."""
        params = _make_params(adjustment_cost_convex=2.0, adjustment_cost_fixed=0.0)
        invest = tf.constant([[1.0]])
        k = tf.constant([[4.0]])
        total, marginal = AdjustmentCostCalculator.calculate(invest, k, params)
        expected_total = 2.0 / 2.0 * 1.0 / 4.0  # 0.25
        expected_marginal = 2.0 * 1.0 / 4.0  # 0.5
        np.testing.assert_allclose(float(total), expected_total, atol=1e-5)
        np.testing.assert_allclose(float(marginal), expected_marginal, atol=1e-5)

    def test_cost_nonnegative(self):
        params = _make_params()
        invest = tf.random.uniform((50, 1), -1, 1)
        k = tf.random.uniform((50, 1), 0.5, 3.0)
        total, _ = AdjustmentCostCalculator.calculate(invest, k, params)
        assert tf.reduce_all(total >= 0)

    def test_with_grad_includes_invest_prob(self):
        params = _make_params(adjustment_cost_fixed=0.05)
        invest = tf.constant([[1.0]])
        k = tf.constant([[2.0]])
        invest_prob = tf.constant([[0.5]])
        total, _ = AdjustmentCostCalculator.calculate_with_grad(invest, k, invest_prob, params)
        # Fixed cost should be 0.05 * 2.0 * 0.5 = 0.05
        assert float(total) > 0


# ─────────────── Cash Flow ───────────────


class TestCashFlowCalculator:
    """Tests for CashFlowCalculator."""

    def test_basic_cash_flow(self):
        profit = tf.constant([[10.0]])
        invest = tf.constant([[3.0]])
        adj_cost = tf.constant([[1.0]])
        cf = CashFlowCalculator.basic_cash_flow(profit, invest, adj_cost)
        np.testing.assert_allclose(float(cf), 6.0, atol=1e-6)

    def test_risky_cash_flow(self):
        payout = tf.constant([[5.0]])
        issuance_cost = tf.constant([[1.0]])
        cf = CashFlowCalculator.risky_cash_flow(payout, issuance_cost)
        np.testing.assert_allclose(float(cf), 4.0, atol=1e-6)


# ─────────────── Bond Pricing ───────────────


class TestBondPricingCalculator:
    """Tests for BondPricingCalculator."""

    def test_recovery_value(self):
        """V_recovery = (1-xi) * (profit' + (1-δ)*K')."""
        params = _make_params(default_cost_proportional=0.3, depreciation_rate=0.1)
        profit = tf.constant([[10.0]])
        k_next = tf.constant([[5.0]])
        rec = BondPricingCalculator.recovery_value(profit, k_next, params)
        expected = 0.7 * (10.0 + 0.9 * 5.0)  # 0.7 * 14.5 = 10.15
        np.testing.assert_allclose(float(rec), expected, atol=1e-4)

    def test_bond_payoff_no_default(self):
        recovery = tf.constant([[100.0]])
        b_face = tf.constant([[50.0]])
        is_default = tf.constant([[0.0]])
        payoff = BondPricingCalculator.bond_payoff(recovery, b_face, is_default)
        np.testing.assert_allclose(float(payoff), 50.0, atol=1e-5)

    def test_bond_payoff_default_recovery_below_face(self):
        recovery = tf.constant([[30.0]])
        b_face = tf.constant([[50.0]])
        is_default = tf.constant([[1.0]])
        payoff = BondPricingCalculator.bond_payoff(recovery, b_face, is_default)
        np.testing.assert_allclose(float(payoff), 30.0, atol=1e-5)

    def test_bond_payoff_default_recovery_above_face(self):
        recovery = tf.constant([[80.0]])
        b_face = tf.constant([[50.0]])
        is_default = tf.constant([[1.0]])
        payoff = BondPricingCalculator.bond_payoff(recovery, b_face, is_default)
        np.testing.assert_allclose(float(payoff), 50.0, atol=1e-5)

    def test_risk_neutral_price_no_default(self):
        """With full repayment, q ≈ 1/(1+r)."""
        r = 0.04
        b_next = tf.constant([[1.0]])
        expected_payoff = b_next  # full repayment
        q = BondPricingCalculator.risk_neutral_price(
            expected_payoff, b_next, r, epsilon_debt=1e-6, min_price=0.01,
        )
        q_rf = 1.0 / (1.0 + r)
        np.testing.assert_allclose(float(q), q_rf, atol=1e-4)

    def test_risk_neutral_price_zero_debt(self):
        """With zero debt, price = q_rf."""
        r = 0.04
        b_next = tf.constant([[0.0]])
        q = BondPricingCalculator.risk_neutral_price(
            tf.constant([[0.0]]), b_next, r, epsilon_debt=1e-6, min_price=0.01,
        )
        q_rf = 1.0 / (1.0 + r)
        np.testing.assert_allclose(float(q), q_rf, atol=1e-4)

    def test_price_has_floor(self):
        """Price never drops below min_price."""
        r = 0.04
        b_next = tf.constant([[100.0]])
        q = BondPricingCalculator.risk_neutral_price(
            tf.constant([[0.01]]), b_next, r, epsilon_debt=1e-6, min_price=0.05,
        )
        assert float(q) >= 0.05 - 1e-6


# ─────────────── Issuance Costs ───────────────


class TestIssuanceCostCalculator:
    """Tests for IssuanceCostCalculator."""

    def test_positive_payout_no_cost(self):
        params = _make_params()
        payout = tf.constant([[5.0]])
        cost = IssuanceCostCalculator.calculate(payout, params)
        np.testing.assert_allclose(float(cost), 0.0, atol=1e-6)

    def test_negative_payout_has_cost(self):
        params = _make_params(equity_issuance_cost_fixed=0.1, equity_issuance_cost_linear=0.05)
        payout = tf.constant([[-3.0]])
        cost = IssuanceCostCalculator.calculate(payout, params)
        expected = (0.1 + 0.05 * 3.0) * 1.0  # is_issuing = 1
        np.testing.assert_allclose(float(cost), expected, atol=1e-5)

    def test_with_grad_differentiable(self):
        params = _make_params(equity_issuance_cost_fixed=0.1, equity_issuance_cost_linear=0.05)
        payout = tf.Variable([[-2.0]])
        with tf.GradientTape() as tape:
            cost = IssuanceCostCalculator.calculate_with_grad(payout, params)
            loss = tf.reduce_sum(cost)
        grad = tape.gradient(loss, payout)
        assert grad is not None


# ─────────────── Debt Flow ───────────────


class TestDebtFlowCalculator:
    """Tests for DebtFlowCalculator."""

    def test_zero_debt_zero_flow(self):
        params = _make_params()
        b_next = tf.constant([[0.0]])
        q = tf.constant([[0.96]])
        debt_inflow, tax_shield = DebtFlowCalculator.calculate(b_next, q, params)
        np.testing.assert_allclose(float(debt_inflow), 0.0, atol=1e-6)
        np.testing.assert_allclose(float(tax_shield), 0.0, atol=1e-6)

    def test_positive_debt_inflow(self):
        params = _make_params(corporate_tax_rate=0.2, risk_free_rate=0.04)
        b_next = tf.constant([[10.0]])
        q = tf.constant([[0.96]])
        debt_inflow, tax_shield = DebtFlowCalculator.calculate(b_next, q, params)
        expected_inflow = 0.96 * 10.0
        np.testing.assert_allclose(float(debt_inflow), expected_inflow, atol=1e-4)
        assert float(tax_shield) > 0  # should have positive tax shield


# ─────────────── Steady State ───────────────


class TestSteadyStateCalculator:
    """Tests for SteadyStateCalculator."""

    def test_positive_capital(self):
        params = _make_params()
        k_ss = SteadyStateCalculator.calculate_capital(params)
        assert k_ss > 0

    def test_higher_beta_higher_capital(self):
        """More patient economy → higher steady-state capital."""
        k_low = SteadyStateCalculator.calculate_capital(_make_params(discount_factor=0.9))
        k_high = SteadyStateCalculator.calculate_capital(_make_params(discount_factor=0.99))
        assert k_high > k_low

    def test_known_formula(self):
        """k_ss = ((1-tau)*theta / (r+delta))^(1/(1-theta))."""
        params = _make_params(
            discount_factor=0.96, capital_share=0.3,
            depreciation_rate=0.1, corporate_tax_rate=0.2,
        )
        r = 1.0 / 0.96 - 1.0
        expected = ((0.8 * 0.3) / (r + 0.1)) ** (1.0 / 0.7)
        k_ss = SteadyStateCalculator.calculate_capital(params)
        np.testing.assert_allclose(k_ss, expected, atol=1e-6)


# ─────────────── Collateral ───────────────


class TestCollateralCalculator:
    """Tests for CollateralCalculator."""

    def test_higher_capital_higher_limit(self):
        params = _make_params()
        k1 = tf.constant([[1.0]])
        k2 = tf.constant([[3.0]])
        lim1 = CollateralCalculator.calculate_limit(k1, 0.5, params)
        lim2 = CollateralCalculator.calculate_limit(k2, 0.5, params)
        assert float(lim2) > float(lim1)

    def test_limit_positive(self):
        params = _make_params()
        k = tf.random.uniform((20, 1), 0.5, 5.0)
        lim = CollateralCalculator.calculate_limit(k, 0.5, params)
        assert tf.reduce_all(lim > 0)

    def test_recovery_fraction_effect(self):
        params = _make_params()
        k = tf.constant([[2.0]])
        lim_low = CollateralCalculator.calculate_limit(k, 0.5, params, recovery_fraction=0.2)
        lim_high = CollateralCalculator.calculate_limit(k, 0.5, params, recovery_fraction=0.8)
        assert float(lim_high) > float(lim_low)
