"""Unit tests for dividend computation functions."""

import pytest
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

from unittest.mock import MagicMock
from econ_models.dl.dividends.basic_dividend import compute_basic_cash_flow
from econ_models.dl.dividends.risky_dividend import (
    compute_invest_dividend,
    compute_no_invest_dividend,
)


def _make_params(**overrides):
    """Create a minimal mock params object with correct attribute names."""
    defaults = dict(
        capital_share=0.3,
        depreciation_rate=0.1,
        adjustment_cost_convex=0.5,
        adjustment_cost_fixed=0.01,
        risk_free_rate=0.02,
        corporate_tax_rate=0.0,
        equity_issuance_cost_fixed=0.0,
        equity_issuance_cost_linear=0.0,
    )
    defaults.update(overrides)
    p = MagicMock()
    for k, v in defaults.items():
        setattr(p, k, v)
    return p


class TestBasicCashFlow:
    """Tests for basic model dividend."""

    def test_output_shape(self):
        params = _make_params()
        k = tf.constant([[1.0], [2.0]])
        z = tf.constant([[1.0], [1.0]])
        k_prime = tf.constant([[1.05], [2.1]])
        invest_prob = tf.constant([[1.0], [1.0]])
        d = compute_basic_cash_flow(k, z, k_prime, invest_prob, params)
        assert d.shape == (2, 1)

    def test_zero_investment_high_dividend(self):
        """No net investment → dividend ≈ output − depreciation replacement."""
        params = _make_params(adjustment_cost_convex=0.0, adjustment_cost_fixed=0.0)
        k = tf.constant([[1.0]])
        z = tf.constant([[1.0]])
        # k_prime exactly covers depreciation
        k_prime = k * (1.0 - params.depreciation_rate)
        invest_prob = tf.constant([[1.0]])
        d = compute_basic_cash_flow(k, z, k_prime, invest_prob, params)
        # output = z * k^alpha, investment = 0 (k_prime = (1-delta)*k)
        # dividend should be positive
        assert float(d.numpy().item()) > 0

    def test_finite_output(self):
        params = _make_params()
        k = tf.random.uniform((4, 1), 0.5, 2.0)
        z = tf.random.uniform((4, 1), 0.8, 1.2)
        k_prime = tf.random.uniform((4, 1), 0.5, 2.0)
        invest_prob = tf.random.uniform((4, 1), 0.0, 1.0)
        d = compute_basic_cash_flow(k, z, k_prime, invest_prob, params)
        assert tf.reduce_all(tf.math.is_finite(d))


class TestRiskyInvestDividend:
    """Tests for risky model invest-path dividend."""

    def test_output_shape(self):
        params = _make_params()
        k = tf.constant([[1.0], [2.0]])
        b = tf.constant([[0.1], [0.2]])
        z = tf.constant([[1.0], [1.0]])
        k_prime = tf.constant([[1.1], [2.1]])
        b_prime = tf.constant([[0.12], [0.22]])
        q = tf.constant([[0.98], [0.97]])
        invest_prob = tf.constant([[1.0], [1.0]])
        d = compute_invest_dividend(k, b, z, k_prime, b_prime, q, invest_prob, params)
        assert d.shape == (2, 1)

    def test_finite_output(self):
        params = _make_params()
        k = tf.random.uniform((4, 1), 0.5, 2.0)
        b = tf.random.uniform((4, 1), 0.0, 0.5)
        z = tf.random.uniform((4, 1), 0.8, 1.2)
        k_prime = tf.random.uniform((4, 1), 0.5, 2.0)
        b_prime = tf.random.uniform((4, 1), 0.0, 0.5)
        q = tf.random.uniform((4, 1), 0.9, 1.0)
        invest_prob = tf.random.uniform((4, 1), 0.0, 1.0)
        d = compute_invest_dividend(k, b, z, k_prime, b_prime, q, invest_prob, params)
        assert tf.reduce_all(tf.math.is_finite(d))


class TestRiskyNoInvestDividend:
    """Tests for risky model no-invest-path dividend."""

    def test_output_shape(self):
        params = _make_params()
        k = tf.constant([[1.0]])
        b = tf.constant([[0.1]])
        z = tf.constant([[1.0]])
        b_prime = tf.constant([[0.12]])
        q = tf.constant([[0.98]])
        d = compute_no_invest_dividend(k, b, z, b_prime, q, params)
        assert d.shape == (1, 1)

    def test_finite_output(self):
        params = _make_params()
        k = tf.random.uniform((4, 1), 0.5, 2.0)
        b = tf.random.uniform((4, 1), 0.0, 0.5)
        z = tf.random.uniform((4, 1), 0.8, 1.2)
        b_prime = tf.random.uniform((4, 1), 0.0, 0.5)
        q = tf.random.uniform((4, 1), 0.9, 1.0)
        d = compute_no_invest_dividend(k, b, z, b_prime, q, params)
        assert tf.reduce_all(tf.math.is_finite(d))
