# tests/unit/test_curriculum.py
"""
Unit tests for curriculum learning utilities.
"""

import unittest

import tensorflow as tf
import numpy as np

from econ_models.core.sampling.curriculum import CurriculumBounds


class TestCurriculumBounds(unittest.TestCase):
    """Test cases for CurriculumBounds."""

    def test_get_bounds_at_zero_progress(self):
        """Test that zero progress gives narrow bounds around center."""
        global_min = tf.constant(0.0)
        global_max = tf.constant(10.0)
        safe_center = tf.constant(5.0)
        progress = tf.constant(0.0)
        initial_ratio = 0.1
        
        curr_min, curr_max = CurriculumBounds.get_curriculum_bounds(
            global_min, global_max, safe_center, progress, initial_ratio
        )
        
        # At progress=0, ratio = 0.1
        # curr_min = 5 - (5 - 0) * 0.1 = 4.5
        # curr_max = 5 + (10 - 5) * 0.1 = 5.5
        self.assertAlmostEqual(curr_min.numpy(), 4.5, places=5)
        self.assertAlmostEqual(curr_max.numpy(), 5.5, places=5)

    def test_get_bounds_at_full_progress(self):
        """Test that full progress gives global bounds."""
        global_min = tf.constant(0.0)
        global_max = tf.constant(10.0)
        safe_center = tf.constant(5.0)
        progress = tf.constant(1.0)
        initial_ratio = 0.1
        
        curr_min, curr_max = CurriculumBounds.get_curriculum_bounds(
            global_min, global_max, safe_center, progress, initial_ratio
        )
        
        self.assertAlmostEqual(curr_min.numpy(), 0.0, places=5)
        self.assertAlmostEqual(curr_max.numpy(), 10.0, places=5)

    def test_get_bounds_at_half_progress(self):
        """Test bounds at 50% progress."""
        global_min = tf.constant(0.0)
        global_max = tf.constant(10.0)
        safe_center = tf.constant(5.0)
        progress = tf.constant(0.5)
        initial_ratio = 0.0
        
        curr_min, curr_max = CurriculumBounds.get_curriculum_bounds(
            global_min, global_max, safe_center, progress, initial_ratio
        )
        
        # At progress=0.5, ratio = 0.5
        # curr_min = 5 - 5 * 0.5 = 2.5
        # curr_max = 5 + 5 * 0.5 = 7.5
        self.assertAlmostEqual(curr_min.numpy(), 2.5, places=5)
        self.assertAlmostEqual(curr_max.numpy(), 7.5, places=5)

    def test_bounds_expand_monotonically(self):
        """Test that bounds expand monotonically with progress."""
        global_min = tf.constant(0.0)
        global_max = tf.constant(10.0)
        safe_center = tf.constant(5.0)
        initial_ratio = 0.1
        
        prev_min, prev_max = 5.0, 5.0
        
        for p in [0.0, 0.25, 0.5, 0.75, 1.0]:
            progress = tf.constant(p)
            curr_min, curr_max = CurriculumBounds.get_curriculum_bounds(
                global_min, global_max, safe_center, progress, initial_ratio
            )
            
            self.assertLessEqual(curr_min.numpy(), prev_min)
            self.assertGreaterEqual(curr_max.numpy(), prev_max)
            
            prev_min = curr_min.numpy()
            prev_max = curr_max.numpy()

    def test_bounds_asymmetric_center(self):
        """Test bounds with asymmetric center point."""
        global_min = tf.constant(0.0)
        global_max = tf.constant(10.0)
        safe_center = tf.constant(2.0)  # Closer to min
        progress = tf.constant(0.5)
        initial_ratio = 0.0
        
        curr_min, curr_max = CurriculumBounds.get_curriculum_bounds(
            global_min, global_max, safe_center, progress, initial_ratio
        )
        
        # curr_min = 2 - (2 - 0) * 0.5 = 1.0
        # curr_max = 2 + (10 - 2) * 0.5 = 6.0
        self.assertAlmostEqual(curr_min.numpy(), 1.0, places=5)
        self.assertAlmostEqual(curr_max.numpy(), 6.0, places=5)

    def test_bounds_with_full_initial_ratio(self):
        """Test that initial_ratio=1.0 gives full bounds immediately."""
        global_min = tf.constant(0.0)
        global_max = tf.constant(10.0)
        safe_center = tf.constant(5.0)
        progress = tf.constant(0.0)
        initial_ratio = 1.0
        
        curr_min, curr_max = CurriculumBounds.get_curriculum_bounds(
            global_min, global_max, safe_center, progress, initial_ratio
        )
        
        self.assertAlmostEqual(curr_min.numpy(), 0.0, places=5)
        self.assertAlmostEqual(curr_max.numpy(), 10.0, places=5)


if __name__ == "__main__":
    unittest.main()