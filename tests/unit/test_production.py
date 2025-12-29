# tests/unit/test_production.py
"""
Unit tests for production function calculations.
"""

import unittest
from unittest.mock import MagicMock

import tensorflow as tf
import numpy as np

from econ_models.econ.production import ProductionFunctions


class TestCobbDouglas(unittest.TestCase):
    """Test cases for Cobb-Douglas production function."""

    def setUp(self):
        """Set up test fixtures."""
        self.params = MagicMock()
        self.params.capital_share = 0.33

    def test_cobb_douglas_scalar_inputs(self):
        """Test Cobb-Douglas with scalar inputs."""
        capital = tf.constant(1.0)
        productivity = tf.constant(1.0)
        
        output = ProductionFunctions.cobb_douglas(capital, productivity, self.params)
        
        # Y = 1.0 * 1.0^0.33 = 1.0
        self.assertAlmostEqual(output.numpy(), 1.0, places=5)

    def test_cobb_douglas_unit_productivity(self):
        """Test that output equals K^theta when Z=1."""
        capital = tf.constant(2.0)
        productivity = tf.constant(1.0)
        
        output = ProductionFunctions.cobb_douglas(capital, productivity, self.params)
        expected = 2.0 ** 0.33
        
        self.assertAlmostEqual(output.numpy(), expected, places=5)

    def test_cobb_douglas_productivity_scales_output(self):
        """Test that productivity scales output linearly."""
        capital = tf.constant(2.0)
        productivity_low = tf.constant(1.0)
        productivity_high = tf.constant(2.0)
        
        output_low = ProductionFunctions.cobb_douglas(
            capital, productivity_low, self.params
        )
        output_high = ProductionFunctions.cobb_douglas(
            capital, productivity_high, self.params
        )
        
        self.assertAlmostEqual(
            output_high.numpy() / output_low.numpy(), 2.0, places=5
        )

    def test_cobb_douglas_zero_capital(self):
        """Test that zero capital produces zero output."""
        capital = tf.constant(0.0)
        productivity = tf.constant(1.0)
        
        output = ProductionFunctions.cobb_douglas(capital, productivity, self.params)
        
        self.assertAlmostEqual(output.numpy(), 0.0, places=5)

    def test_cobb_douglas_batch_inputs(self):
        """Test Cobb-Douglas with batch inputs."""
        capital = tf.constant([1.0, 2.0, 4.0])
        productivity = tf.constant([1.0, 1.0, 1.0])
        
        output = ProductionFunctions.cobb_douglas(capital, productivity, self.params)
        
        self.assertEqual(output.shape, (3,))
        # Check monotonicity
        output_np = output.numpy()
        self.assertLess(output_np[0], output_np[1])
        self.assertLess(output_np[1], output_np[2])

    def test_cobb_douglas_different_capital_shares(self):
        """Test with different capital share parameters."""
        capital = tf.constant(4.0)
        productivity = tf.constant(1.0)
        
        self.params.capital_share = 0.5
        output_half = ProductionFunctions.cobb_douglas(
            capital, productivity, self.params
        )
        
        # Y = 4^0.5 = 2.0
        self.assertAlmostEqual(output_half.numpy(), 2.0, places=5)


class TestCalculateInvestment(unittest.TestCase):
    """Test cases for investment calculation."""

    def setUp(self):
        """Set up test fixtures."""
        self.params = MagicMock()
        self.params.depreciation_rate = 0.1

    def test_investment_no_growth(self):
        """Test investment when capital is maintained at same level."""
        k_curr = tf.constant(10.0)
        k_next = tf.constant(10.0)
        
        investment = ProductionFunctions.calculate_investment(
            k_curr, k_next, self.params
        )
        
        # I = 10 - 0.9 * 10 = 1.0 (replace depreciated capital)
        self.assertAlmostEqual(investment.numpy(), 1.0, places=5)

    def test_investment_capital_growth(self):
        """Test investment when capital grows."""
        k_curr = tf.constant(10.0)
        k_next = tf.constant(12.0)
        
        investment = ProductionFunctions.calculate_investment(
            k_curr, k_next, self.params
        )
        
        # I = 12 - 0.9 * 10 = 3.0
        self.assertAlmostEqual(investment.numpy(), 3.0, places=5)

    def test_investment_capital_decline(self):
        """Test investment when capital shrinks (negative investment)."""
        k_curr = tf.constant(10.0)
        k_next = tf.constant(8.0)
        
        investment = ProductionFunctions.calculate_investment(
            k_curr, k_next, self.params
        )
        
        # I = 8 - 0.9 * 10 = -1.0
        self.assertAlmostEqual(investment.numpy(), -1.0, places=5)

    def test_investment_zero_capital(self):
        """Test investment starting from zero capital."""
        k_curr = tf.constant(0.0)
        k_next = tf.constant(5.0)
        
        investment = ProductionFunctions.calculate_investment(
            k_curr, k_next, self.params
        )
        
        self.assertAlmostEqual(investment.numpy(), 5.0, places=5)

    def test_investment_batch_inputs(self):
        """Test investment calculation with batch inputs."""
        k_curr = tf.constant([10.0, 20.0, 30.0])
        k_next = tf.constant([10.0, 20.0, 30.0])
        
        investment = ProductionFunctions.calculate_investment(
            k_curr, k_next, self.params
        )
        
        expected = [1.0, 2.0, 3.0]  # Just replacing depreciation
        np.testing.assert_array_almost_equal(investment.numpy(), expected, decimal=5)

    def test_investment_full_depreciation(self):
        """Test investment with 100% depreciation rate."""
        self.params.depreciation_rate = 1.0
        k_curr = tf.constant(10.0)
        k_next = tf.constant(10.0)
        
        investment = ProductionFunctions.calculate_investment(
            k_curr, k_next, self.params
        )
        
        # I = 10 - 0 * 10 = 10.0
        self.assertAlmostEqual(investment.numpy(), 10.0, places=5)


if __name__ == "__main__":
    unittest.main()