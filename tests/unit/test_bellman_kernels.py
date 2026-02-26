"""Unit tests for bellman_kernels: compute_ev, bellman_update, sup_norm_diff.

All tests run on CPU — no GPU required.
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf

# Force CPU for CI
tf.config.set_visible_devices([], 'GPU')

import pytest

from econ_models.vfi.kernels.bellman_kernels import (
    bellman_update,
    compute_ev,
    sup_norm_diff,
)


class TestComputeEV:
    """Tests for compute_ev."""

    def test_shape_2d(self):
        """Output shape matches input for 2-D value function."""
        v = tf.constant([[1.0, 2.0], [3.0, 4.0]])  # (2, 2)
        P = tf.constant([[0.8, 0.2], [0.3, 0.7]])  # (2, 2)
        beta = tf.constant(0.96)
        result = compute_ev(v, P, beta)
        assert result.shape == v.shape

    def test_shape_3d(self):
        """Output shape matches input for 3-D value function."""
        v = tf.ones((3, 4, 2))
        P = tf.constant([[0.8, 0.2], [0.3, 0.7]])
        beta = tf.constant(0.96)
        result = compute_ev(v, P, beta)
        assert result.shape == v.shape

    def test_known_values_2d(self):
        """Verify against manual NumPy computation for 2-D case."""
        v_np = np.array([[1.0, 2.0], [3.0, 4.0]])
        P_np = np.array([[0.8, 0.2], [0.3, 0.7]])
        beta_val = 0.96
        expected = beta_val * (v_np @ P_np.T)

        v = tf.constant(v_np, dtype=tf.float32)
        P = tf.constant(P_np, dtype=tf.float32)
        beta = tf.constant(beta_val, dtype=tf.float32)
        result = compute_ev(v, P, beta).numpy()
        np.testing.assert_allclose(result, expected, atol=1e-5)

    def test_known_values_3d(self):
        """Verify against manual NumPy computation for 3-D case."""
        nk, nb, nz = 2, 3, 2
        v_np = np.random.rand(nk, nb, nz).astype(np.float32)
        P_np = np.array([[0.8, 0.2], [0.3, 0.7]], dtype=np.float32)
        beta_val = 0.96
        expected = beta_val * np.tensordot(v_np, P_np, axes=[[2], [1]])

        v = tf.constant(v_np)
        P = tf.constant(P_np)
        beta = tf.constant(beta_val, dtype=tf.float32)
        result = compute_ev(v, P, beta).numpy()
        np.testing.assert_allclose(result, expected, atol=1e-5)

    def test_zero_beta(self):
        """With beta=0, expected value should be all zeros."""
        v = tf.ones((3, 4, 2))
        P = tf.constant([[0.5, 0.5], [0.5, 0.5]])
        beta = tf.constant(0.0)
        result = compute_ev(v, P, beta).numpy()
        np.testing.assert_allclose(result, 0.0, atol=1e-7)


class TestBellmanUpdate:
    """Tests for bellman_update."""

    def test_adjust_dominates(self):
        """When adjust > wait > 0, v_next = v_adjust."""
        v_adjust = tf.constant([[10.0, 20.0]])
        v_wait = tf.constant([[5.0, 15.0]])
        v_curr = tf.constant([[0.0, 0.0]])
        v_next, diff = bellman_update(v_adjust, v_wait, v_curr)
        np.testing.assert_allclose(v_next.numpy(), v_adjust.numpy())
        assert float(diff) > 0

    def test_wait_dominates(self):
        """When wait > adjust > 0, v_next = v_wait."""
        v_adjust = tf.constant([[5.0, 15.0]])
        v_wait = tf.constant([[10.0, 20.0]])
        v_curr = tf.constant([[0.0, 0.0]])
        v_next, diff = bellman_update(v_adjust, v_wait, v_curr)
        np.testing.assert_allclose(v_next.numpy(), v_wait.numpy())

    def test_default_floor(self):
        """When both branches are negative, default floor (0) applies."""
        v_adjust = tf.constant([[-5.0, -10.0]])
        v_wait = tf.constant([[-3.0, -8.0]])
        v_curr = tf.constant([[0.0, 0.0]])
        v_next, diff = bellman_update(v_adjust, v_wait, v_curr)
        np.testing.assert_allclose(v_next.numpy(), 0.0, atol=1e-7)

    def test_diff_is_zero_for_converged(self):
        """When v_next == v_curr, diff should be zero."""
        v = tf.constant([[10.0, 20.0]])
        v_next, diff = bellman_update(v, v, v)
        assert float(diff) == pytest.approx(0.0, abs=1e-7)


class TestSupNormDiff:
    """Tests for sup_norm_diff."""

    def test_identical_tensors(self):
        """Identical tensors should have zero difference."""
        a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        result = sup_norm_diff(a, a)
        assert float(result) == pytest.approx(0.0, abs=1e-7)

    def test_known_difference(self):
        """Difference between tensors that differ by a known amount."""
        a = tf.constant([[1.0, 2.0]])
        b = tf.constant([[1.5, 2.3]])
        result = sup_norm_diff(a, b)
        assert float(result) == pytest.approx(0.5, abs=1e-5)

    def test_negative_values(self):
        """Works correctly with negative values."""
        a = tf.constant([[-10.0, 5.0]])
        b = tf.constant([[-7.0, 5.0]])
        result = sup_norm_diff(a, b)
        assert float(result) == pytest.approx(3.0, abs=1e-5)
