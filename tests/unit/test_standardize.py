# tests/unit/test_standardize.py
"""
Unit tests for state space normalization.
"""

import unittest
from unittest.mock import MagicMock

import tensorflow as tf
import numpy as np

from econ_models.core.standardize import StateSpaceNormalizer


class TestStateSpaceNormalizerInit(unittest.TestCase):
    """Test cases for StateSpaceNormalizer initialization."""

    def test_init_with_valid_config(self):
        """Test initialization with valid configuration."""
        config = MagicMock()
        config.capital_min = 0.1
        config.capital_max = 10.0
        config.productivity_min = 0.5
        config.productivity_max = 2.0
        config.debt_min = 0.0
        config.debt_max = 5.0
        
        normalizer = StateSpaceNormalizer(config)
        
        self.assertAlmostEqual(normalizer.k_min.numpy(), 0.1, places=5)
        self.assertAlmostEqual(normalizer.k_range.numpy(), 9.9, places=5)

    def test_init_raises_error_without_capital_bounds(self):
        """Test that missing capital bounds raises ValueError."""
        config = MagicMock()
        config.capital_min = None
        config.capital_max = 10.0
        config.productivity_min = 0.5
        config.productivity_max = 2.0
        
        with self.assertRaises(ValueError) as context:
            StateSpaceNormalizer(config)
        
        self.assertIn("Capital boundaries", str(context.exception))

    def test_init_raises_error_without_productivity_bounds(self):
        """Test that missing productivity bounds raises ValueError."""
        config = MagicMock()
        config.capital_min = 0.1
        config.capital_max = 10.0
        config.productivity_min = None
        config.productivity_max = 2.0
        
        with self.assertRaises(ValueError) as context:
            StateSpaceNormalizer(config)
        
        self.assertIn("Productivity boundaries", str(context.exception))

    def test_init_without_debt_bounds_uses_defaults(self):
        """Test that missing debt bounds uses default values."""
        config = MagicMock()
        config.capital_min = 0.1
        config.capital_max = 10.0
        config.productivity_min = 0.5
        config.productivity_max = 2.0
        config.debt_min = None
        config.debt_max = None
        
        normalizer = StateSpaceNormalizer(config)
        
        self.assertAlmostEqual(normalizer.b_min.numpy(), 0.0, places=5)
        self.assertAlmostEqual(normalizer.b_range.numpy(), 1.0, places=5)


class TestCapitalNormalization(unittest.TestCase):
    """Test cases for capital normalization."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = MagicMock()
        self.config.capital_min = 1.0
        self.config.capital_max = 11.0
        self.config.productivity_min = 0.5
        self.config.productivity_max = 1.5
        self.config.debt_min = 0.0
        self.config.debt_max = 5.0
        self.normalizer = StateSpaceNormalizer(self.config)

    def test_normalize_capital_min_to_zero(self):
        """Test that minimum capital normalizes to 0."""
        k = tf.constant(1.0)
        k_norm = self.normalizer.normalize_capital(k)
        self.assertAlmostEqual(k_norm.numpy(), 0.0, places=5)

    def test_normalize_capital_max_to_one(self):
        """Test that maximum capital normalizes to 1."""
        k = tf.constant(11.0)
        k_norm = self.normalizer.normalize_capital(k)
        self.assertAlmostEqual(k_norm.numpy(), 1.0, places=5)

    def test_normalize_capital_midpoint_to_half(self):
        """Test that midpoint normalizes to 0.5."""
        k = tf.constant(6.0)  # Midpoint of [1, 11]
        k_norm = self.normalizer.normalize_capital(k)
        self.assertAlmostEqual(k_norm.numpy(), 0.5, places=5)

    def test_denormalize_capital_zero_to_min(self):
        """Test that 0 denormalizes to minimum capital."""
        k_norm = tf.constant(0.0)
        k = self.normalizer.denormalize_capital(k_norm)
        self.assertAlmostEqual(k.numpy(), 1.0, places=5)

    def test_denormalize_capital_one_to_max(self):
        """Test that 1 denormalizes to maximum capital."""
        k_norm = tf.constant(1.0)
        k = self.normalizer.denormalize_capital(k_norm)
        self.assertAlmostEqual(k.numpy(), 11.0, places=5)

    def test_normalize_denormalize_roundtrip(self):
        """Test that normalizing then denormalizing returns original."""
        k_original = tf.constant(5.0)
        k_norm = self.normalizer.normalize_capital(k_original)
        k_recovered = self.normalizer.denormalize_capital(k_norm)
        self.assertAlmostEqual(k_recovered.numpy(), 5.0, places=5)

    def test_normalize_capital_batch(self):
        """Test normalization with batch inputs."""
        k = tf.constant([1.0, 6.0, 11.0])
        k_norm = self.normalizer.normalize_capital(k)
        expected = [0.0, 0.5, 1.0]
        np.testing.assert_array_almost_equal(k_norm.numpy(), expected, decimal=5)


class TestProductivityNormalization(unittest.TestCase):
    """Test cases for productivity normalization."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = MagicMock()
        self.config.capital_min = 1.0
        self.config.capital_max = 11.0
        self.config.productivity_min = 0.5
        self.config.productivity_max = 1.5
        self.config.debt_min = 0.0
        self.config.debt_max = 5.0
        self.normalizer = StateSpaceNormalizer(self.config)

    def test_normalize_productivity_min_to_zero(self):
        """Test that minimum productivity normalizes to 0."""
        z = tf.constant(0.5)
        z_norm = self.normalizer.normalize_productivity(z)
        self.assertAlmostEqual(z_norm.numpy(), 0.0, places=5)

    def test_normalize_productivity_max_to_one(self):
        """Test that maximum productivity normalizes to 1."""
        z = tf.constant(1.5)
        z_norm = self.normalizer.normalize_productivity(z)
        self.assertAlmostEqual(z_norm.numpy(), 1.0, places=5)

    def test_normalize_productivity_batch(self):
        """Test productivity normalization with batch inputs."""
        z = tf.constant([0.5, 1.0, 1.5])
        z_norm = self.normalizer.normalize_productivity(z)
        expected = [0.0, 0.5, 1.0]
        np.testing.assert_array_almost_equal(z_norm.numpy(), expected, decimal=5)


class TestDebtNormalization(unittest.TestCase):
    """Test cases for debt normalization."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = MagicMock()
        self.config.capital_min = 1.0
        self.config.capital_max = 11.0
        self.config.productivity_min = 0.5
        self.config.productivity_max = 1.5
        self.config.debt_min = 0.0
        self.config.debt_max = 10.0
        self.normalizer = StateSpaceNormalizer(self.config)

    def test_normalize_debt_min_to_zero(self):
        """Test that minimum debt normalizes to 0."""
        b = tf.constant(0.0)
        b_norm = self.normalizer.normalize_debt(b)
        self.assertAlmostEqual(b_norm.numpy(), 0.0, places=5)

    def test_normalize_debt_max_to_one(self):
        """Test that maximum debt normalizes to 1."""
        b = tf.constant(10.0)
        b_norm = self.normalizer.normalize_debt(b)
        self.assertAlmostEqual(b_norm.numpy(), 1.0, places=5)

    def test_denormalize_debt_roundtrip(self):
        """Test debt normalize-denormalize roundtrip."""
        b_original = tf.constant(5.0)
        b_norm = self.normalizer.normalize_debt(b_original)
        b_recovered = self.normalizer.denormalize_debt(b_norm)
        self.assertAlmostEqual(b_recovered.numpy(), 5.0, places=5)


if __name__ == "__main__":
    unittest.main()