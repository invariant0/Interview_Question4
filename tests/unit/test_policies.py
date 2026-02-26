"""Unit tests for policies.py: extract_basic_policies and extract_risky_policies."""

from __future__ import annotations

import numpy as np
import tensorflow as tf

tf.config.set_visible_devices([], 'GPU')

import pytest

from econ_models.vfi.policies import extract_basic_policies, extract_risky_policies


class TestExtractBasicPolicies:
    """Tests for extract_basic_policies."""

    def test_identity_indices(self):
        """Indices equal to range(n) should return the grid back."""
        k_grid = tf.constant([0.1, 0.5, 1.0, 2.0])
        idx = tf.constant([[0, 1], [2, 3], [1, 0], [3, 2]])  # (4, 2)
        result = extract_basic_policies(k_grid, idx).numpy()
        expected = np.array([
            [0.1, 0.5], [1.0, 2.0], [0.5, 0.1], [2.0, 1.0]
        ])
        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_constant_policy(self):
        """When all indices point to the same value."""
        k_grid = tf.constant([0.5, 1.0, 1.5])
        idx = tf.constant([[1, 1], [1, 1], [1, 1]])
        result = extract_basic_policies(k_grid, idx).numpy()
        np.testing.assert_allclose(result, 1.0, atol=1e-6)

    def test_shape(self):
        """Output shape matches the index shape."""
        nk, nz = 5, 3
        k_grid = tf.linspace(0.1, 2.0, nk)
        idx = tf.zeros((nk, nz), dtype=tf.int32)
        result = extract_basic_policies(k_grid, idx)
        assert result.shape == (nk, nz)


class TestExtractRiskyPolicies:
    """Tests for extract_risky_policies."""

    def test_keys(self):
        """Output dict contains all expected keys."""
        nk, nb, nz = 3, 4, 2
        k_grid = tf.linspace(0.1, 2.0, nk)
        b_grid = tf.linspace(0.0, 1.0, nb)

        result = extract_risky_policies(
            k_grid, b_grid,
            v_adjust_vals=tf.ones((nk, nb, nz)),
            v_wait_vals=tf.zeros((nk, nb, nz)),
            policy_adjust_idx_flat=tf.zeros((nk, nb, nz), tf.int32),
            policy_b_wait_idx=tf.zeros((nk, nb, nz), tf.int32),
            n_debt=nb,
            v_default_eps=0.01,
        )
        expected_keys = {
            "policy_default", "policy_adjust",
            "policy_k_idx", "policy_b_idx", "policy_b_wait_idx",
            "policy_k_values", "policy_b_values", "policy_b_wait_values",
        }
        assert set(result.keys()) == expected_keys

    def test_default_masking(self):
        """Firms in default region get index=-1 and NaN values."""
        nk, nb, nz = 2, 2, 2
        k_grid = tf.constant([1.0, 2.0])
        b_grid = tf.constant([0.0, 0.5])

        # Both branches return negative value → default
        v_adjust = tf.fill((nk, nb, nz), -5.0)
        v_wait = tf.fill((nk, nb, nz), -5.0)
        idx_flat = tf.zeros((nk, nb, nz), tf.int32)
        idx_bw = tf.zeros((nk, nb, nz), tf.int32)

        result = extract_risky_policies(
            k_grid, b_grid, v_adjust, v_wait,
            idx_flat, idx_bw,
            n_debt=nb,
            v_default_eps=0.01,
        )

        assert np.all(result["policy_default"].numpy())
        assert np.all(result["policy_k_idx"].numpy() == -1)
        assert np.all(result["policy_b_idx"].numpy() == -1)
        assert np.all(np.isnan(result["policy_k_values"].numpy()))
        assert np.all(np.isnan(result["policy_b_values"].numpy()))

    def test_no_default_region(self):
        """When all values are positive, no defaults."""
        nk, nb, nz = 2, 3, 2
        k_grid = tf.constant([1.0, 2.0])
        b_grid = tf.constant([0.0, 0.5, 1.0])

        v_adjust = tf.fill((nk, nb, nz), 10.0)
        v_wait = tf.fill((nk, nb, nz), 5.0)
        idx_flat = tf.zeros((nk, nb, nz), tf.int32)
        idx_bw = tf.zeros((nk, nb, nz), tf.int32)

        result = extract_risky_policies(
            k_grid, b_grid, v_adjust, v_wait,
            idx_flat, idx_bw,
            n_debt=nb,
            v_default_eps=0.01,
        )

        assert not np.any(result["policy_default"].numpy())
        assert np.all(result["policy_k_idx"].numpy() >= 0)
        assert np.all(~np.isnan(result["policy_k_values"].numpy()))

    def test_adjust_vs_wait_indicator(self):
        """When adjust > wait, policy_adjust is True and vice versa."""
        nk, nb, nz = 2, 2, 2
        k_grid = tf.constant([1.0, 2.0])
        b_grid = tf.constant([0.0, 0.5])

        # Adjust dominates in first half, wait in second
        v_adjust = tf.constant([[[10.0, 5.0], [10.0, 5.0]],
                                 [[10.0, 5.0], [10.0, 5.0]]])
        v_wait = tf.constant([[[5.0, 10.0], [5.0, 10.0]],
                               [[5.0, 10.0], [5.0, 10.0]]])
        idx_flat = tf.zeros((nk, nb, nz), tf.int32)
        idx_bw = tf.zeros((nk, nb, nz), tf.int32)

        result = extract_risky_policies(
            k_grid, b_grid, v_adjust, v_wait,
            idx_flat, idx_bw,
            n_debt=nb,
            v_default_eps=0.01,
        )
        pa = result["policy_adjust"].numpy()
        # z=0: adjust dominates (10 > 5), z=1: wait dominates (5 < 10)
        assert np.all(pa[:, :, 0] == True)
        assert np.all(pa[:, :, 1] == False)

    def test_flat_index_decomposition(self):
        """Flat index correctly decomposed into k and b indices."""
        nk, nb, nz = 3, 4, 2
        k_grid = tf.linspace(0.1, 2.0, nk)
        b_grid = tf.linspace(0.0, 1.0, nb)

        # flat_idx = k_idx * nb + b_idx
        # e.g., flat_idx=5 → k_idx=1, b_idx=1 when nb=4
        flat_idx = tf.fill((nk, nb, nz), 5)  # k_idx=1, b_idx=1
        v_adjust = tf.fill((nk, nb, nz), 10.0)
        v_wait = tf.fill((nk, nb, nz), 1.0)
        idx_bw = tf.zeros((nk, nb, nz), tf.int32)

        result = extract_risky_policies(
            k_grid, b_grid, v_adjust, v_wait,
            flat_idx, idx_bw,
            n_debt=nb,
            v_default_eps=0.01,
        )

        np.testing.assert_array_equal(result["policy_k_idx"].numpy(), 1)
        np.testing.assert_array_equal(result["policy_b_idx"].numpy(), 1)
