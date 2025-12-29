# tests/unit/test_math.py
"""
Unit tests for backward compatibility wrapper.
"""

import unittest
from unittest.mock import MagicMock, patch

import tensorflow as tf

from econ_models.core.math import MathUtils


class TestMathUtilsBackwardCompatibility(unittest.TestCase):
    """Test that MathUtils properly delegates to specialized modules."""

    def test_log_ar1_transition_is_accessible(self):
        """Test that log_ar1_transition is accessible via MathUtils."""
        z_curr = tf.constant(1.0)
        rho = 0.9
        epsilon = tf.constant(0.0)
        
        result = MathUtils.log_ar1_transition(z_curr, rho, epsilon)
        
        self.assertAlmostEqual(result.numpy(), 1.0, places=5)

    def test_get_curriculum_bounds_is_accessible(self):
        """Test that get_curriculum_bounds is accessible via MathUtils."""
        global_min = tf.constant(0.0)
        global_max = tf.constant(10.0)
        safe_center = tf.constant(5.0)
        progress = tf.constant(1.0)
        initial_ratio = 0.1
        
        curr_min, curr_max = MathUtils.get_curriculum_bounds(
            global_min, global_max, safe_center, progress, initial_ratio
        )
        
        self.assertAlmostEqual(curr_min.numpy(), 0.0, places=5)
        self.assertAlmostEqual(curr_max.numpy(), 10.0, places=5)

    def test_sample_heavy_tailed_vector(self):
        """Test sample_heavy_tailed_vector function."""
        minval = tf.constant(0.0)
        maxval = tf.constant(10.0)
        
        samples = MathUtils.sample_heavy_tailed_vector(
            shape=(100,), minval=minval, maxval=maxval
        )
        
        self.assertEqual(samples.shape, (100,))
        self.assertTrue(all(samples.numpy() >= 0.0))
        self.assertTrue(all(samples.numpy() <= 10.0))


if __name__ == "__main__":
    unittest.main()