# tests/unit/test_transitions.py
"""
Unit tests for stochastic transition functions.
"""

import unittest

import tensorflow as tf
import numpy as np

from econ_models.core.sampling.transitions import TransitionFunctions


class TestLogAR1Transition(unittest.TestCase):
    """Test cases for log-AR(1) transition function."""

    def test_transition_unit_productivity_zero_shock(self):
        """Test that z=1 with zero shock stays at 1."""
        z_curr = tf.constant(1.0)
        rho = 0.9
        epsilon = tf.constant(0.0)
        
        z_next = TransitionFunctions.log_ar1_transition(z_curr, rho, epsilon)
        
        # ln(1) = 0, rho * 0 + 0 = 0, exp(0) = 1
        self.assertAlmostEqual(z_next.numpy(), 1.0, places=5)

    def test_transition_positive_shock_increases_productivity(self):
        """Test that positive shock increases next period productivity."""
        z_curr = tf.constant(1.0)
        rho = 0.9
        epsilon = tf.constant(0.1)
        
        z_next = TransitionFunctions.log_ar1_transition(z_curr, rho, epsilon)
        
        self.assertGreater(z_next.numpy(), 1.0)

    def test_transition_negative_shock_decreases_productivity(self):
        """Test that negative shock decreases next period productivity."""
        z_curr = tf.constant(1.0)
        rho = 0.9
        epsilon = tf.constant(-0.1)
        
        z_next = TransitionFunctions.log_ar1_transition(z_curr, rho, epsilon)
        
        self.assertLess(z_next.numpy(), 1.0)

    def test_transition_persistence_effect(self):
        """Test that higher rho maintains more of current level."""
        z_curr = tf.constant(2.0)  # Above steady state
        epsilon = tf.constant(0.0)
        
        z_high_rho = TransitionFunctions.log_ar1_transition(
            z_curr, rho=0.99, epsilon=epsilon
        )
        z_low_rho = TransitionFunctions.log_ar1_transition(
            z_curr, rho=0.5, epsilon=epsilon
        )
        
        # Higher persistence keeps z closer to current value
        self.assertGreater(z_high_rho.numpy(), z_low_rho.numpy())

    def test_transition_formula(self):
        """Test the log-AR(1) formula directly."""
        z_curr = tf.constant(2.0)
        rho = 0.8
        epsilon = tf.constant(0.05)
        
        z_next = TransitionFunctions.log_ar1_transition(z_curr, rho, epsilon)
        
        # Manual calculation
        ln_z = np.log(2.0)
        ln_z_prime = 0.8 * ln_z + 0.05
        expected = np.exp(ln_z_prime)
        
        self.assertAlmostEqual(z_next.numpy(), expected, places=5)

    def test_transition_always_positive(self):
        """Test that output is always positive."""
        z_curr = tf.constant(0.5)
        rho = 0.9
        epsilon = tf.constant(-0.5)
        
        z_next = TransitionFunctions.log_ar1_transition(z_curr, rho, epsilon)
        
        self.assertGreater(z_next.numpy(), 0.0)

    def test_transition_handles_small_z(self):
        """Test that very small z values are handled safely."""
        z_curr = tf.constant(1e-10)
        rho = 0.9
        epsilon = tf.constant(0.0)
        
        z_next = TransitionFunctions.log_ar1_transition(z_curr, rho, epsilon)
        
        self.assertTrue(np.isfinite(z_next.numpy()))
        self.assertGreater(z_next.numpy(), 0.0)

    def test_transition_batch_inputs(self):
        """Test transition with batch inputs."""
        z_curr = tf.constant([0.5, 1.0, 2.0])
        rho = 0.9
        epsilon = tf.constant([0.0, 0.0, 0.0])
        
        z_next = TransitionFunctions.log_ar1_transition(z_curr, rho, epsilon)
        
        self.assertEqual(z_next.shape, (3,))
        # All should be positive
        self.assertTrue(np.all(z_next.numpy() > 0))

    def test_transition_zero_persistence(self):
        """Test with zero persistence (iid shocks)."""
        z_curr = tf.constant(2.0)
        rho = 0.0
        epsilon = tf.constant(0.0)
        
        z_next = TransitionFunctions.log_ar1_transition(z_curr, rho, epsilon)
        
        # With rho=0 and epsilon=0: ln(z') = 0, z' = 1
        self.assertAlmostEqual(z_next.numpy(), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()