# tests/unit/test_adjustment_costs.py
"""
Unit tests for adjustment cost calculations.
"""

import unittest
from unittest.mock import MagicMock

import tensorflow as tf
import numpy as np

from econ_models.econ.adjustment_costs import AdjustmentCostCalculator


class TestAdjustmentCostCalculator(unittest.TestCase):
    """Test cases for AdjustmentCostCalculator."""

    def setUp(self):
        """Set up test fixtures."""
        self.params = MagicMock()
        self.params.adjustment_cost_convex = 0.5
        self.params.adjustment_cost_fixed = 0.02

    def test_calculate_zero_investment_zero_cost(self):
        """Test that zero investment has zero convex cost."""
        investment = tf.constant(0.0)
        capital = tf.constant(10.0)
        
        total_cost, marginal_cost = AdjustmentCostCalculator.calculate(
            investment, capital, self.params
        )
        
        # Fixed cost is also zero when investment is zero
        self.assertAlmostEqual(total_cost.numpy(), 0.0, places=5)
        self.assertAlmostEqual(marginal_cost.numpy(), 0.0, places=5)

    def test_calculate_convex_cost_formula(self):
        """Test convex cost formula: psi_0 * I^2 / (2*K)."""
        investment = tf.constant(2.0)
        capital = tf.constant(10.0)
        
        # Set fixed cost to zero to isolate convex cost
        self.params.adjustment_cost_fixed = 0.0
        
        total_cost, _ = AdjustmentCostCalculator.calculate(
            investment, capital, self.params
        )
        
        # cost = 0.5 * 4 / 20 = 0.1
        self.assertAlmostEqual(total_cost.numpy(), 0.1, places=5)

    def test_calculate_fixed_cost_formula(self):
        """Test fixed cost formula: psi_1 * K when I != 0."""
        investment = tf.constant(2.0)
        capital = tf.constant(10.0)
        
        # Set convex cost to zero to isolate fixed cost
        self.params.adjustment_cost_convex = 0.0
        
        total_cost, _ = AdjustmentCostCalculator.calculate(
            investment, capital, self.params
        )
        
        # cost = 0.02 * 10 = 0.2
        self.assertAlmostEqual(total_cost.numpy(), 0.2, places=5)

    def test_calculate_marginal_cost_formula(self):
        """Test marginal cost formula: psi_0 * I / K."""
        investment = tf.constant(2.0)
        capital = tf.constant(10.0)
        
        _, marginal_cost = AdjustmentCostCalculator.calculate(
            investment, capital, self.params
        )
        
        # marginal = 0.5 * 2 / 10 = 0.1
        self.assertAlmostEqual(marginal_cost.numpy(), 0.1, places=5)

    def test_calculate_total_cost_combines_components(self):
        """Test that total cost is sum of convex and fixed costs."""
        investment = tf.constant(2.0)
        capital = tf.constant(10.0)
        
        total_cost, _ = AdjustmentCostCalculator.calculate(
            investment, capital, self.params
        )
        
        # convex = 0.5 * 4 / 20 = 0.1
        # fixed = 0.02 * 10 = 0.2
        # total = 0.3
        self.assertAlmostEqual(total_cost.numpy(), 0.3, places=5)

    def test_calculate_negative_investment(self):
        """Test costs with negative investment (disinvestment)."""
        investment = tf.constant(-2.0)
        capital = tf.constant(10.0)
        
        total_cost, marginal_cost = AdjustmentCostCalculator.calculate(
            investment, capital, self.params
        )
        
        # Convex cost: 0.5 * 4 / 20 = 0.1 (squared, so positive)
        # Fixed cost: 0.02 * 10 = 0.2 (non-zero investment)
        self.assertAlmostEqual(total_cost.numpy(), 0.3, places=5)
        # Marginal cost: 0.5 * (-2) / 10 = -0.1
        self.assertAlmostEqual(marginal_cost.numpy(), -0.1, places=5)

    def test_calculate_cost_increases_quadratically_with_investment(self):
        """Test that convex cost increases with investment squared."""
        capital = tf.constant(10.0)
        self.params.adjustment_cost_fixed = 0.0  # Isolate convex component
        
        cost_1, _ = AdjustmentCostCalculator.calculate(
            tf.constant(1.0), capital, self.params
        )
        cost_2, _ = AdjustmentCostCalculator.calculate(
            tf.constant(2.0), capital, self.params
        )
        
        # Doubling investment should quadruple cost
        self.assertAlmostEqual(
            cost_2.numpy() / cost_1.numpy(), 4.0, places=4
        )

    def test_calculate_handles_small_capital(self):
        """Test that small capital doesn't cause division by zero."""
        investment = tf.constant(1.0)
        capital = tf.constant(1e-6)
        
        total_cost, marginal_cost = AdjustmentCostCalculator.calculate(
            investment, capital, self.params
        )
        
        # Should not raise and should return finite values
        self.assertTrue(np.isfinite(total_cost.numpy()))
        self.assertTrue(np.isfinite(marginal_cost.numpy()))

    def test_calculate_batch_inputs(self):
        """Test adjustment costs with batch inputs."""
        investment = tf.constant([0.0, 1.0, 2.0])
        capital = tf.constant([10.0, 10.0, 10.0])
        
        total_cost, marginal_cost = AdjustmentCostCalculator.calculate(
            investment, capital, self.params
        )
        
        self.assertEqual(total_cost.shape, (3,))
        self.assertEqual(marginal_cost.shape, (3,))
        
        # First element should be zero (no investment)
        self.assertAlmostEqual(total_cost.numpy()[0], 0.0, places=5)


if __name__ == "__main__":
    unittest.main()