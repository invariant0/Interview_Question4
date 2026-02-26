"""Unit tests for continuous critic loss."""

import pytest
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

from econ_models.dl.losses.continuous_critic import compute_continuous_loss


class TestContinuousLoss:
    """Tests for the continuous-action Bellman (AiO) loss."""

    def test_output_is_scalar(self):
        v = tf.constant([[1.0], [2.0]])
        t1 = tf.constant([[0.5], [1.5]])
        t2 = tf.constant([[0.5], [1.5]])
        loss = compute_continuous_loss(v, t1, t2)
        assert loss.shape == ()

    def test_zero_when_perfect(self):
        v = tf.constant([[3.0], [4.0]])
        loss = compute_continuous_loss(v, v, v)
        tf.debugging.assert_near(loss, 0.0, atol=1e-6)

    def test_gradient_through_v(self):
        v = tf.Variable([[1.0]])
        t1 = tf.constant([[0.0]])
        t2 = tf.constant([[0.0]])
        with tf.GradientTape() as tape:
            loss = compute_continuous_loss(v, t1, t2)
        grad = tape.gradient(loss, v)
        assert grad is not None
