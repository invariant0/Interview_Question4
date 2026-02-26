"""Unit tests for default supervision loss."""

import pytest
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

from econ_models.dl.losses.default_supervision import compute_default_loss


class TestDefaultLoss:
    """Tests for default policy supervision."""

    def test_output_is_scalar(self):
        v_continuation = tf.constant([[5.0], [-1.0]])
        default_probs = tf.constant([[0.3], [0.8]])
        loss = compute_default_loss(v_continuation, default_probs, epsilon_threshold=0.0)
        assert loss.shape == ()

    def test_label_computation_negative_v(self):
        """When v_cont < epsilon, label should be 1 (default)."""
        v_continuation = tf.constant([[-1.0]])
        default_probs = tf.constant([[0.5]])
        loss = compute_default_loss(v_continuation, default_probs, epsilon_threshold=0.0)
        # Label = 1.0 (default), prob = 0.5 → BCE(0.5, 1.0) = -log(0.5) ≈ 0.693
        tf.debugging.assert_near(loss, 0.693, atol=0.05)

    def test_label_computation_positive_v(self):
        """When v_cont > epsilon, label should be 0 (no default)."""
        v_continuation = tf.constant([[10.0]])
        default_probs = tf.constant([[0.5]])
        loss = compute_default_loss(v_continuation, default_probs, epsilon_threshold=0.0)
        # Label = 0.0 (no default), prob = 0.5 → BCE(0.5, 0.0) = -log(0.5) ≈ 0.693
        tf.debugging.assert_near(loss, 0.693, atol=0.05)

    def test_gradient_flows(self):
        probs = tf.Variable([[0.5], [0.3]])
        v_cont = tf.constant([[5.0], [-1.0]])
        with tf.GradientTape() as tape:
            loss = compute_default_loss(v_cont, probs, epsilon_threshold=0.0)
        grad = tape.gradient(loss, probs)
        assert grad is not None
