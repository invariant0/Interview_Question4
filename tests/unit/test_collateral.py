# tests/unit/test_collateral.py
"""
Unit tests for collateral constraint calculations.
"""

import unittest
from unittest.mock import MagicMock

import tensorflow as tf
import numpy as np

from econ_models.econ.collateral import CollateralCalculator


class TestCollateralCalculator(unittest.TestCase):
    """Test cases for CollateralCalculator."""

    def setUp(self):
        """Set up test fixtures."""
        self.params = MagicMock()
        self.params.corporate_tax_rate = 0.2
        self.params.capital_share = 0.33
        self.params.depreciation_rate = 0.1

    def test_calculate_limit_positive_capital(self):
        """Test collateral limit is positive for positive capital."""
        k_next = tf.constant(10.0)
        z_min = 0.8
        
        limit = CollateralCalculator.calculate_limit(
            k_next, z_min, self.params, recovery_fraction=0.5
        )
        
        self.assertGreater(limit.numpy(), 0.0)

    def test_calculate_limit_zero_capital(self):
        """Test collateral limit is zero for zero capital."""
        k_next = tf.constant(0.0)
        z_min = 0.8
        
        limit = CollateralCalculator.calculate_limit(
            k_next, z_min, self.params, recovery_fraction=0.5
        )
        
        self.assertAlmostEqual(limit.numpy(), 0.0, places=5)

    def test_calculate_limit_known_values(self):
        """Test collateral limit with manually calculated values."""
        k_next = tf.constant(1.0)
        z_min = 1.0
        tau = 0.2
        theta = 0.33
        delta = 0.1
        s = 0.5
        
        # term_1 = (1 - 0.2) * 1.0 * 1.0^0.33 = 0.8
        # term_2 = 0.2 * 0.1 * 1.0 = 0.02
        # term_3 = 0.5 * 1.0 = 0.5
        # total = 1.32
        
        limit = CollateralCalculator.calculate_limit(
            k_next, z_min, self.params, recovery_fraction=s
        )
        
        expected = 0.8 + 0.02 + 0.5
        self.assertAlmostEqual(limit.numpy(), expected, places=5)

    def test_calculate_limit_higher_recovery_increases_limit(self):
        """Test that higher recovery fraction increases borrowing limit."""
        k_next = tf.constant(10.0)
        z_min = 0.8
        
        limit_low = CollateralCalculator.calculate_limit(
            k_next, z_min, self.params, recovery_fraction=0.3
        )
        limit_high = CollateralCalculator.calculate_limit(
            k_next, z_min, self.params, recovery_fraction=0.7
        )
        
        self.assertGreater(limit_high.numpy(), limit_low.numpy())

    def test_calculate_limit_higher_capital_increases_limit(self):
        """Test that more capital increases borrowing limit."""
        z_min = 0.8
        
        limit_low = CollateralCalculator.calculate_limit(
            tf.constant(5.0), z_min, self.params
        )
        limit_high = CollateralCalculator.calculate_limit(
            tf.constant(10.0), z_min, self.params
        )
        
        self.assertGreater(limit_high.numpy(), limit_low.numpy())

    def test_calculate_limit_batch_inputs(self):
        """Test collateral calculation with batch inputs."""
        k_next = tf.constant([1.0, 2.0, 3.0])
        z_min = 0.8
        
        limit = CollateralCalculator.calculate_limit(
            k_next, z_min, self.params
        )
        
        self.assertEqual(limit.shape, (3,))
        # Check monotonicity
        limit_np = limit.numpy()
        self.assertLess(limit_np[0], limit_np[1])
        self.assertLess(limit_np[1], limit_np[2])

    def test_calculate_limit_lower_z_min_decreases_limit(self):
        """Test that lower minimum productivity decreases borrowing limit."""
        k_next = tf.constant(10.0)
        
        limit_high_z = CollateralCalculator.calculate_limit(
            k_next, z_min=1.0, params=self.params
        )
        limit_low_z = CollateralCalculator.calculate_limit(
            k_next, z_min=0.5, params=self.params
        )
        
        self.assertGreater(limit_high_z.numpy(), limit_low_z.numpy())


if __name__ == "__main__":
    unittest.main()