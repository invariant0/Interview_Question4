"""Unit tests for investment decision losses."""

import pytest
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

from econ_models.dl.losses.investment_decision import (
    compute_investment_bce_loss,
    compute_investment_entropy,
    compute_bernoulli_entropy,
)


class TestInvestmentBCE:
    """Tests for investment BCE loss."""

    def test_perfect_prediction(self):
        probs = tf.constant([[0.99], [0.01]])
        labels = tf.constant([[1.0], [0.0]])
        loss = compute_investment_bce_loss(labels, probs)
        assert float(loss) < 0.2  # loose bound for numerical BCE

    def test_output_is_scalar(self):
        probs = tf.random.uniform((8, 1), 0.01, 0.99)
        labels = tf.round(probs)
        loss = compute_investment_bce_loss(labels, probs)
        assert loss.shape == ()

    def test_gradient_flows(self):
        probs = tf.Variable([[0.5], [0.7]])
        labels = tf.constant([[1.0], [0.0]])
        with tf.GradientTape() as tape:
            loss = compute_investment_bce_loss(labels, probs)
        grad = tape.gradient(loss, probs)
        assert grad is not None


class TestInvestmentEntropy:
    """Tests for investment entropy."""

    def test_max_entropy_at_half(self):
        probs = tf.constant([[0.5], [0.5]])
        ent = compute_investment_entropy(probs)
        # -log(0.5) ≈ 0.693
        tf.debugging.assert_near(ent, 0.693, atol=0.01)

    def test_low_entropy_at_extremes(self):
        probs = tf.constant([[0.999], [0.001]])
        ent = compute_investment_entropy(probs)
        assert float(ent) < 0.1


class TestBernoulliEntropy:
    """Tests for Bernoulli entropy."""

    def test_max_entropy_at_half(self):
        probs = tf.constant([[0.5], [0.5]])
        ent = compute_bernoulli_entropy(probs)
        tf.debugging.assert_near(ent, 0.693, atol=0.01)

    def test_per_element_output(self):
        """compute_bernoulli_entropy returns per-element (not scalar)."""
        probs = tf.random.uniform((4, 1), 0.01, 0.99)
        ent = compute_bernoulli_entropy(probs)
        assert ent.shape == (4, 1)
