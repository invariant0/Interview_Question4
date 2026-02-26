"""Unit tests for TransitionFunctions (core/sampling/transitions.py)."""

from __future__ import annotations

import numpy as np
import tensorflow as tf

tf.config.set_visible_devices([], 'GPU')

import pytest

from econ_models.core.sampling.transitions import TransitionFunctions


class TestLogAR1Transition:
    """Tests for log_ar1_transition: ln(z') = rho*ln(z) + eps."""

    def test_zero_shock_persistence(self):
        """With eps=0, z' = z^rho."""
        z = tf.constant([[2.0]])
        rho = 0.9
        eps = tf.constant([[0.0]])
        z_prime = TransitionFunctions.log_ar1_transition(z, rho, eps)
        expected = np.exp(rho * np.log(2.0))
        np.testing.assert_allclose(float(z_prime), expected, atol=1e-5)

    def test_unit_productivity_zero_shock(self):
        """z=1 with eps=0 gives z'=1 (ln(1)=0)."""
        z = tf.constant([[1.0]])
        z_prime = TransitionFunctions.log_ar1_transition(z, 0.9, tf.constant([[0.0]]))
        np.testing.assert_allclose(float(z_prime), 1.0, atol=1e-5)

    def test_positive_shock_increases(self):
        """Positive epsilon raises productivity."""
        z = tf.constant([[1.0]])
        eps_pos = tf.constant([[0.1]])
        eps_zero = tf.constant([[0.0]])
        z_high = TransitionFunctions.log_ar1_transition(z, 0.9, eps_pos)
        z_base = TransitionFunctions.log_ar1_transition(z, 0.9, eps_zero)
        assert float(z_high) > float(z_base)

    def test_negative_shock_decreases(self):
        """Negative epsilon lowers productivity."""
        z = tf.constant([[1.0]])
        eps_neg = tf.constant([[-0.1]])
        eps_zero = tf.constant([[0.0]])
        z_low = TransitionFunctions.log_ar1_transition(z, 0.9, eps_neg)
        z_base = TransitionFunctions.log_ar1_transition(z, 0.9, eps_zero)
        assert float(z_low) < float(z_base)

    def test_output_always_positive(self):
        """z' = exp(...) is always positive."""
        z = tf.random.uniform((100, 1), 0.1, 5.0)
        eps = tf.random.normal((100, 1), 0.0, 0.5)
        z_prime = TransitionFunctions.log_ar1_transition(z, 0.9, eps)
        assert tf.reduce_all(z_prime > 0)

    def test_batch_shape(self):
        """Output shape matches input shape."""
        z = tf.random.uniform((16, 1), 0.5, 2.0)
        eps = tf.random.normal((16, 1))
        z_prime = TransitionFunctions.log_ar1_transition(z, 0.8, eps)
        assert z_prime.shape == (16, 1)

    def test_rho_zero_iid(self):
        """With rho=0, z' = exp(eps) — independent of z_curr."""
        z1 = tf.constant([[0.5]])
        z2 = tf.constant([[5.0]])
        eps = tf.constant([[0.1]])
        z1_prime = TransitionFunctions.log_ar1_transition(z1, 0.0, eps)
        z2_prime = TransitionFunctions.log_ar1_transition(z2, 0.0, eps)
        np.testing.assert_allclose(float(z1_prime), float(z2_prime), atol=1e-5)

    def test_rho_one_random_walk(self):
        """With rho=1, ln(z') = ln(z) + eps → z' = z*exp(eps)."""
        z = tf.constant([[2.0]])
        eps = tf.constant([[0.05]])
        z_prime = TransitionFunctions.log_ar1_transition(z, 1.0, eps)
        expected = 2.0 * np.exp(0.05)
        np.testing.assert_allclose(float(z_prime), expected, atol=1e-5)
