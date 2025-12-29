# tests/unit/test_candidate_sampler.py
"""
Unit tests for candidate action sampling.
"""

import unittest
from unittest.mock import MagicMock

import tensorflow as tf
import numpy as np

from econ_models.core.sampling.candidate_sampler import CandidateSampler


class TestCandidateSampler(unittest.TestCase):
    """Test cases for CandidateSampler."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = MagicMock()
        self.config.capital_min = 0.5
        self.config.capital_max = 5.0
        self.config.capital_steady_state = 2.0
        self.config.debt_min = 0.0
        self.config.debt_max = 3.0
        self.config.curriculum_initial_ratio = 0.1
        
        self.batch_size = 10
        self.n_candidates = 50
        self.k_current = tf.constant([[2.0]] * self.batch_size)
        self.b_current = tf.constant([[1.0]] * self.batch_size)

    def test_sample_candidate_returns_correct_shapes(self):
        """Test that output shapes are correct."""
        k_cand, b_cand = CandidateSampler.sample_candidate(
            batch_size=self.batch_size,
            n_candidates=self.n_candidates,
            k_current=self.k_current,
            b_current=self.b_current,
            config=self.config,
            progress=tf.constant(1.0)
        )
        
        self.assertEqual(k_cand.shape[0], self.batch_size)
        self.assertEqual(b_cand.shape[0], self.batch_size)

    def test_sample_candidate_capital_within_bounds(self):
        """Test that capital candidates are within bounds."""
        k_cand, _ = CandidateSampler.sample_candidate(
            batch_size=self.batch_size,
            n_candidates=self.n_candidates,
            k_current=self.k_current,
            b_current=self.b_current,
            config=self.config,
            progress=tf.constant(1.0)
        )
        
        self.assertTrue(np.all(k_cand.numpy() >= self.config.capital_min))
        self.assertTrue(np.all(k_cand.numpy() <= self.config.capital_max))

    def test_sample_candidate_debt_within_bounds(self):
        """Test that debt candidates are within bounds."""
        _, b_cand = CandidateSampler.sample_candidate(
            batch_size=self.batch_size,
            n_candidates=self.n_candidates,
            k_current=self.k_current,
            b_current=self.b_current,
            config=self.config,
            progress=tf.constant(1.0)
        )
        
        self.assertTrue(np.all(b_cand.numpy() >= self.config.debt_min))
        self.assertTrue(np.all(b_cand.numpy() <= self.config.debt_max))

    def test_sample_candidate_candidates_scale_with_progress(self):
        """Test that number of candidates scales with progress."""
        _, b_early = CandidateSampler.sample_candidate(
            batch_size=self.batch_size,
            n_candidates=100,
            k_current=self.k_current,
            b_current=self.b_current,
            config=self.config,
            progress=tf.constant(0.0)
        )
        _, b_late = CandidateSampler.sample_candidate(
            batch_size=self.batch_size,
            n_candidates=100,
            k_current=self.k_current,
            b_current=self.b_current,
            config=self.config,
            progress=tf.constant(1.0)
        )
        
        self.assertLessEqual(b_early.shape[1], b_late.shape[1])


class TestNormalizeProgress(unittest.TestCase):
    """Test cases for progress normalization."""

    def test_normalize_none_returns_one(self):
        """Test that None progress returns 1.0."""
        result = CandidateSampler._normalize_progress(None)
        self.assertAlmostEqual(result.numpy(), 1.0, places=5)

    def test_normalize_tensor_returns_same_value(self):
        """Test that tensor progress is passed through."""
        result = CandidateSampler._normalize_progress(tf.constant(0.5))
        self.assertAlmostEqual(result.numpy(), 0.5, places=5)


class TestScaleCandidates(unittest.TestCase):
    """Test cases for candidate scaling."""

    def test_scale_at_full_progress(self):
        """Test that full progress gives full candidates."""
        result = CandidateSampler._scale_candidates(100, tf.constant(1.0))
        self.assertEqual(result.numpy(), 100)

    def test_scale_at_zero_progress(self):
        """Test that zero progress gives reduced candidates."""
        result = CandidateSampler._scale_candidates(100, tf.constant(0.0))
        self.assertLess(result.numpy(), 100)
        self.assertGreaterEqual(result.numpy(), 16)  # Minimum is 16

    def test_scale_minimum_candidates(self):
        """Test that minimum candidates is enforced."""
        result = CandidateSampler._scale_candidates(10, tf.constant(0.0))
        self.assertGreaterEqual(result.numpy(), 16)


if __name__ == "__main__":
    unittest.main()