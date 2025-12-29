# tests/unit/test_bond_pricing.py
"""
Unit tests for bond pricing calculations.
"""

import unittest
from unittest.mock import MagicMock

import tensorflow as tf
import numpy as np

from econ_models.econ.bond_pricing import BondPricingCalculator


class TestRecoveryValue(unittest.TestCase):
    """Test cases for recovery value calculation."""

    def setUp(self):
        """Set up test fixtures."""
        self.params = MagicMock()
        self.params.default_cost_proportional = 0.2
        self.params.depreciation_rate = 0.1

    def test_recovery_value_positive(self):
        """Test recovery value is positive for positive inputs."""
        profit_next = tf.constant(5.0)
        capital_next = tf.constant(10.0)
        
        recovery = BondPricingCalculator.recovery_value(
            profit_next, capital_next, self.params
        )
        
        self.assertGreater(recovery.numpy(), 0.0)

    def test_recovery_value_formula(self):
        """Test recovery value formula: (1 - xi) * (profit + (1-delta)*K)."""
        profit_next = tf.constant(5.0)
        capital_next = tf.constant(10.0)
        
        recovery = BondPricingCalculator.recovery_value(
            profit_next, capital_next, self.params
        )
        
        # V = (1 - 0.2) * (5 + 0.9 * 10) = 0.8 * 14 = 11.2
        self.assertAlmostEqual(recovery.numpy(), 11.2, places=5)

    def test_recovery_value_higher_default_cost_reduces_recovery(self):
        """Test that higher default cost reduces recovery value."""
        profit_next = tf.constant(5.0)
        capital_next = tf.constant(10.0)
        
        recovery_low = BondPricingCalculator.recovery_value(
            profit_next, capital_next, self.params
        )
        
        self.params.default_cost_proportional = 0.5
        recovery_high = BondPricingCalculator.recovery_value(
            profit_next, capital_next, self.params
        )
        
        self.assertLess(recovery_high.numpy(), recovery_low.numpy())

    def test_recovery_value_zero_capital_and_profit(self):
        """Test recovery value with zero inputs."""
        profit_next = tf.constant(0.0)
        capital_next = tf.constant(0.0)
        
        recovery = BondPricingCalculator.recovery_value(
            profit_next, capital_next, self.params
        )
        
        self.assertAlmostEqual(recovery.numpy(), 0.0, places=5)


class TestBondPayoff(unittest.TestCase):
    """Test cases for bond payoff calculation."""

    def test_bond_payoff_no_default_full_repayment(self):
        """Test that no default results in full face value payment."""
        recovery_val = tf.constant(15.0)
        debt_face = tf.constant(10.0)
        is_default = tf.constant(0.0)
        
        payoff = BondPricingCalculator.bond_payoff(
            recovery_val, debt_face, is_default
        )
        
        self.assertAlmostEqual(payoff.numpy(), 10.0, places=5)

    def test_bond_payoff_default_recovery_less_than_face(self):
        """Test that default with low recovery pays recovery value."""
        recovery_val = tf.constant(5.0)
        debt_face = tf.constant(10.0)
        is_default = tf.constant(1.0)
        
        payoff = BondPricingCalculator.bond_payoff(
            recovery_val, debt_face, is_default
        )
        
        self.assertAlmostEqual(payoff.numpy(), 5.0, places=5)

    def test_bond_payoff_default_recovery_exceeds_face(self):
        """Test that default recovery is capped at face value."""
        recovery_val = tf.constant(15.0)
        debt_face = tf.constant(10.0)
        is_default = tf.constant(1.0)
        
        payoff = BondPricingCalculator.bond_payoff(
            recovery_val, debt_face, is_default
        )
        
        # Bondholders receive min(15, 10) = 10
        self.assertAlmostEqual(payoff.numpy(), 10.0, places=5)

    def test_bond_payoff_zero_debt(self):
        """Test payoff with zero debt face value."""
        recovery_val = tf.constant(10.0)
        debt_face = tf.constant(0.0)
        is_default = tf.constant(0.0)
        
        payoff = BondPricingCalculator.bond_payoff(
            recovery_val, debt_face, is_default
        )
        
        self.assertAlmostEqual(payoff.numpy(), 0.0, places=5)

    def test_bond_payoff_batch_inputs(self):
        """Test bond payoff with batch inputs."""
        recovery_val = tf.constant([5.0, 15.0, 8.0])
        debt_face = tf.constant([10.0, 10.0, 10.0])
        is_default = tf.constant([1.0, 1.0, 0.0])
        
        payoff = BondPricingCalculator.bond_payoff(
            recovery_val, debt_face, is_default
        )
        
        expected = [5.0, 10.0, 10.0]
        np.testing.assert_array_almost_equal(payoff.numpy(), expected, decimal=5)


class TestRiskNeutralPrice(unittest.TestCase):
    """Test cases for risk-neutral bond pricing."""

    def test_risk_neutral_price_no_default_risk(self):
        """Test that zero default risk gives risk-free price."""
        expected_payoff = tf.constant(10.0)
        debt_next = tf.constant(10.0)
        
        price = BondPricingCalculator.risk_neutral_price(
            expected_payoff=expected_payoff,
            debt_next=debt_next,
            risk_free_rate=0.04,
            epsilon_debt=0.001,
            min_price=0.1
        )
        
        # q = 10 / (1.04 * 10) = 1/1.04 ≈ 0.9615
        self.assertAlmostEqual(price.numpy(), 1.0 / 1.04, places=4)

    def test_risk_neutral_price_with_default_risk(self):
        """Test that default risk reduces bond price."""
        expected_payoff = tf.constant(8.0)  # Expected < face value
        debt_next = tf.constant(10.0)
        
        price = BondPricingCalculator.risk_neutral_price(
            expected_payoff=expected_payoff,
            debt_next=debt_next,
            risk_free_rate=0.04,
            epsilon_debt=0.001,
            min_price=0.1
        )
        
        # q = 8 / (1.04 * 10) = 0.769
        expected_price = 8.0 / (1.04 * 10.0)
        self.assertAlmostEqual(price.numpy(), expected_price, places=4)

    def test_risk_neutral_price_zero_debt_returns_risk_free(self):
        """Test that zero debt returns risk-free price."""
        expected_payoff = tf.constant(0.0)
        debt_next = tf.constant(0.0)
        
        price = BondPricingCalculator.risk_neutral_price(
            expected_payoff=expected_payoff,
            debt_next=debt_next,
            risk_free_rate=0.04,
            epsilon_debt=0.001,
            min_price=0.1
        )
        
        # Should return risk-free price for zero debt
        self.assertAlmostEqual(price.numpy(), 1.0 / 1.04, places=4)

    def test_risk_neutral_price_respects_minimum(self):
        """Test that price is bounded by minimum."""
        expected_payoff = tf.constant(0.5)  # Very low payoff
        debt_next = tf.constant(10.0)
        min_price = 0.3
        
        price = BondPricingCalculator.risk_neutral_price(
            expected_payoff=expected_payoff,
            debt_next=debt_next,
            risk_free_rate=0.04,
            epsilon_debt=0.001,
            min_price=min_price
        )
        
        self.assertGreaterEqual(price.numpy(), min_price)

    def test_risk_neutral_price_respects_maximum(self):
        """Test that price is bounded by risk-free price."""
        expected_payoff = tf.constant(15.0)  # Very high payoff
        debt_next = tf.constant(10.0)
        
        price = BondPricingCalculator.risk_neutral_price(
            expected_payoff=expected_payoff,
            debt_next=debt_next,
            risk_free_rate=0.04,
            epsilon_debt=0.001,
            min_price=0.1
        )
        
        # Price should not exceed risk-free price
        self.assertLessEqual(price.numpy(), 1.0 / 1.04 + 0.001)


if __name__ == "__main__":
    unittest.main()