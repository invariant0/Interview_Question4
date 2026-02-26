"""Unit tests for loss functions."""

import pytest
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

from econ_models.dl.losses.bellman import compute_bellman_loss
from econ_models.dl.losses.value_objective import compute_value_maximization_loss
from econ_models.dl.losses.fischer_burmeister import compute_fb_value_loss


class TestBellmanLoss:
    """Tests for the AiO double-sample Bellman loss."""

    def test_known_values(self):
        v = tf.constant([[1.0]])
        t1 = tf.constant([[0.5]])
        t2 = tf.constant([[0.5]])
        loss = compute_bellman_loss(v, t1, t2)
        # mean((1.0-0.5) * (1.0-0.5)) = 0.25
        tf.debugging.assert_near(loss, 0.25, atol=1e-6)

    def test_zero_residual(self):
        v = tf.constant([[2.0], [3.0]])
        loss = compute_bellman_loss(v, v, v)
        tf.debugging.assert_near(loss, 0.0, atol=1e-6)

    def test_output_is_scalar(self):
        v = tf.random.uniform((8, 1))
        t1 = tf.random.uniform((8, 1))
        t2 = tf.random.uniform((8, 1))
        loss = compute_bellman_loss(v, t1, t2)
        assert loss.shape == ()

    def test_gradient_flows_through_v(self):
        v = tf.Variable([[1.0], [2.0]])
        t1 = tf.constant([[0.5], [1.5]])
        t2 = tf.constant([[0.5], [1.5]])
        with tf.GradientTape() as tape:
            loss = compute_bellman_loss(v, t1, t2)
        grad = tape.gradient(loss, v)
        assert grad is not None
        assert tf.reduce_all(tf.math.is_finite(grad))


class TestValueMaximizationLoss:
    """Tests for policy value-maximization loss."""

    def test_known_values(self):
        rhs1 = tf.constant([[2.0]])
        rhs2 = tf.constant([[4.0]])
        loss = compute_value_maximization_loss(rhs1, rhs2)
        # -mean((2+4)/2) = -3.0
        tf.debugging.assert_near(loss, -3.0, atol=1e-6)

    def test_higher_rhs_produces_lower_loss(self):
        rhs_small = tf.constant([[1.0], [1.0]])
        rhs_large = tf.constant([[10.0], [10.0]])
        loss_small = compute_value_maximization_loss(rhs_small, rhs_small)
        loss_large = compute_value_maximization_loss(rhs_large, rhs_large)
        assert float(loss_large) < float(loss_small)


class TestFBValueLoss:
    """Tests for Fischer-Burmeister value loss."""

    def test_output_is_scalar(self):
        v_value = tf.constant([[1.0], [2.0]])
        v_cont = tf.constant([[0.5], [1.5]])
        loss = compute_fb_value_loss(v_value, v_cont)
        assert loss.shape == ()

    def test_loss_is_nonnegative(self):
        v_value = tf.random.uniform((8, 1), -1, 3)
        v_cont = tf.random.uniform((8, 1), -1, 3)
        loss = compute_fb_value_loss(v_value, v_cont)
        assert float(loss) >= 0.0
