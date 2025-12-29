
# tests/unit/test_state_sampler.py
"""
Unit tests for state variable sampling.
"""

import unittest
from unittest.mock import MagicMock

import tensorflow as tf
import numpy as np

from econ_models.core.sampling.state_sampler import StateSampler


class TestStateSampler(unittest.TestCase):
    """Test cases for StateSampler."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = MagicMock()
        self.config.capital_min = 0.5
        self.config.capital_max = 5.0
        self.config.capital_steady_state = 2.0
        self.config.productivity_min = 0.5
        self.config.productivity_max = 2.0
        self.config.debt_min = 0.0
        self.config.debt_max = 3.0
        self.config.curriculum_initial_ratio = 0.1

    def test_sample_states_returns_two_tensors_without_debt(self):
        """Test that sampling without debt returns K and Z."""
        k, z = StateSampler.sample_states(
            batch_size=100,
            config=self.config,
            include_debt=False,
            progress=tf.constant(1.0)
        )
        
        self.assertEqual(len(k.shape), 2)
        self.assertEqual(k.shape[1], 1)
        self.assertEqual(len(z.shape), 2)
        self.assertEqual(z.shape[1], 1)

    def test_sample_states_returns_three_tensors_with_debt(self):
        """Test that sampling with debt returns K, B, and Z."""
        k, b, z = StateSampler.sample_states(
            batch_size=100,
            config=self.config,
            include_debt=True,
            progress=tf.constant(1.0)
        )
        
        self.assertEqual(len(k.shape), 2)
        self.assertEqual(len(b.shape), 2)
        self.assertEqual(len(z.shape), 2)

    def test_sample_states_capital_within_bounds(self):
        """Test that sampled capital is within configured bounds."""
        k, _ = StateSampler.sample_states(
            batch_size=1000,
            config=self.config,
            include_debt=False,
            progress=tf.constant(1.0)
        )
        
        self.assertTrue(np.all(k.numpy() >= self.config.capital_min))
        self.assertTrue(np.all(k.numpy() <= self.config.capital_max))

    def test_sample_states_productivity_within_bounds(self):
        """Test that sampled productivity is within configured bounds."""
        _, z = StateSampler.sample_states(
            batch_size=1000,
            config=self.config,
            include_debt=False,
            progress=tf.constant(1.0)
        )
        
        self.assertTrue(np.all(z.numpy() >= self.config.productivity_min))
        self.assertTrue(np.all(z.numpy() <= self.config.productivity_max))

    def test_sample_states_debt_within_bounds(self):
        """Test that sampled debt is within configured bounds."""
        _, b, _ = StateSampler.sample_states(
            batch_size=1000,
            config=self.config,
            include_debt=True,
            progress=tf.constant(1.0)
        )
        
        self.assertTrue(np.all(b.numpy() >= self.config.debt_min))
        self.assertTrue(np.all(b.numpy() <= self.config.debt_max))

    def test_sample_states_batch_size_scales_with_progress(self):
        """Test that batch size increases with progress."""
        max_batch = 100
        
        k_early, _ = StateSampler.sample_states(
            batch_size=max_batch,
            config=self.config,
            include_debt=False,
            progress=tf.constant(0.0)
        )
        k_late, _ = StateSampler.sample_states(
            batch_size=max_batch,
            config=self.config,
            include_debt=False,
            progress=tf.constant(1.0)
        )
        
        self.assertLessEqual(k_early.shape[0], k_late.shape[0])

    def test_sample_states_none_progress_defaults_to_full(self):
        """Test that None progress defaults to full progress."""
        k, _ = StateSampler.sample_states(
            batch_size=100,
            config=self.config,
            include_debt=False,
            progress=None
        )
        
        # Should use full domain
        self.assertGreater(k.shape[0], 0)


class TestSampleBetaScaled(unittest.TestCase):
    """Test cases for beta-scaled sampling."""

    def test_samples_within_range(self):
        """Test that samples are within specified range."""
        minval = tf.constant(0.0)
        maxval = tf.constant(10.0)
        
        samples = StateSampler._sample_beta_scaled(
            shape=(1000,), minval=minval, maxval=maxval
        )
        
        self.assertTrue(np.all(samples.numpy() >= 0.0))
        self.assertTrue(np.all(samples.numpy() <= 10.0))

    def test_uniform_sampling_with_alpha_beta_one(self):
        """Test that alpha=beta=1 gives approximately uniform distribution."""
        minval = tf.constant(0.0)
        maxval = tf.constant(1.0)
        
        samples = StateSampler._sample_beta_scaled(
            shape=(10000,), minval=minval, maxval=maxval,
            alpha=1.0, beta=1.0
        )
        
        # Mean should be approximately 0.5 for uniform
        mean = np.mean(samples.numpy())
        self.assertAlmostEqual(mean, 0.5, places=1)

    def test_correct_output_shape(self):
        """Test that output shape matches requested shape."""
        minval = tf.constant(0.0)
        maxval = tf.constant(1.0)
        
        samples = StateSampler._sample_beta_scaled(
            shape=(50,), minval=minval, maxval=maxval
        )
        
        self.assertEqual(samples.shape, (50,))


if __name__ == "__main__":
    unittest.main()