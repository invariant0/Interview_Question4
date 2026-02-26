"""Unit tests for FischerBurmeisterLoss (core/math.py)."""

from __future__ import annotations

import numpy as np
import tensorflow as tf

tf.config.set_visible_devices([], 'GPU')

import pytest

from econ_models.core.math import FischerBurmeisterLoss


class TestFBFunction:
    """Tests for fb_function: Φ(a,b) = a + b - sqrt(a² + b² + ε)."""

    def test_both_zero(self):
        """Φ(0,0) ≈ 0."""
        result = FischerBurmeisterLoss.fb_function(
            tf.constant(0.0), tf.constant(0.0)
        )
        assert abs(float(result)) < 1e-3

    def test_both_positive(self):
        """Φ(a,b) < 0 when both a,b > 0 (complementarity violated)."""
        result = FischerBurmeisterLoss.fb_function(
            tf.constant(3.0), tf.constant(4.0)
        )
        expected = 3.0 + 4.0 - np.sqrt(9.0 + 16.0)  # 7 - 5 = 2
        np.testing.assert_allclose(float(result), expected, atol=1e-3)

    def test_one_zero(self):
        """Φ(a,0) ≈ a - |a| ≈ 0 for a > 0."""
        result = FischerBurmeisterLoss.fb_function(
            tf.constant(5.0), tf.constant(0.0)
        )
        assert abs(float(result)) < 1e-3

    def test_batch_shape(self):
        """Works with batched tensors."""
        a = tf.constant([[1.0], [2.0], [3.0]])
        b = tf.constant([[4.0], [5.0], [6.0]])
        result = FischerBurmeisterLoss.fb_function(a, b)
        assert result.shape == (3, 1)

    def test_negative_values(self):
        """Φ handles negative inputs."""
        result = FischerBurmeisterLoss.fb_function(
            tf.constant(-2.0), tf.constant(3.0)
        )
        expected = -2.0 + 3.0 - np.sqrt(4.0 + 9.0)
        np.testing.assert_allclose(float(result), expected, atol=1e-3)


class TestComplementarityResidual:
    """Tests for complementarity_residual(v_value, v_cont)."""

    def test_v_equals_vcont(self):
        """When V = Vcont > 0, gap = 0, so Φ(V, 0) ≈ 0."""
        v = tf.constant([[5.0]])
        result = FischerBurmeisterLoss.complementarity_residual(v, v)
        assert abs(float(result)) < 1e-3

    def test_v_zero_vcont_negative(self):
        """When V = 0 and Vcont < 0, gap > 0, so Φ(0, gap) ≈ 0."""
        v = tf.constant([[0.0]])
        vcont = tf.constant([[-3.0]])
        result = FischerBurmeisterLoss.complementarity_residual(v, vcont)
        # gap = 0 - (-3) = 3; Φ(0, 3) = 0 + 3 - sqrt(0+9) = 0
        assert abs(float(result)) < 1e-3


class TestComputeLoss:
    """Tests for compute_loss."""

    def test_output_is_scalar(self):
        v_value = tf.constant([[1.0], [2.0]])
        v_cont = tf.constant([[0.5], [1.5]])
        loss = FischerBurmeisterLoss.compute_loss(v_value, v_cont)
        assert loss.shape == ()

    def test_loss_nonnegative(self):
        v_value = tf.random.uniform((8, 1), -1, 3)
        v_cont = tf.random.uniform((8, 1), -1, 3)
        loss = FischerBurmeisterLoss.compute_loss(v_value, v_cont)
        assert float(loss) >= 0.0

    def test_perfect_complementarity_zero_loss(self):
        """When V = max(0, Vcont), loss should be near zero."""
        v_cont = tf.constant([[5.0], [-2.0], [0.0]])
        v_value = tf.maximum(v_cont, 0.0)
        loss = FischerBurmeisterLoss.compute_loss(v_value, v_cont)
        assert float(loss) < 1e-3
