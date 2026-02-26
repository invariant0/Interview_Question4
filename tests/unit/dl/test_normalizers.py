"""Unit tests for StateSpaceNormalizer and ParamSpaceNormalizer (core/standardize.py)."""

from __future__ import annotations

import numpy as np
import tensorflow as tf

tf.config.set_visible_devices([], 'GPU')

import pytest

from econ_models.core.standardize import StateSpaceNormalizer, ParamSpaceNormalizer
from econ_models.config.dl_config import DeepLearningConfig


def _make_dl_config(**overrides):
    defaults = dict(
        capital_min=0.5, capital_max=3.0,
        productivity_min=0.5, productivity_max=1.5,
        debt_min=0.0, debt_max=2.0,
        hidden_layers=(32,), activation_function='relu',
    )
    defaults.update(overrides)
    return DeepLearningConfig(**defaults)


class TestStateSpaceNormalizer:
    """Tests for min-max normalisation of state variables."""

    def test_normalize_capital_endpoints(self):
        config = _make_dl_config()
        norm = StateSpaceNormalizer(config)
        assert float(norm.normalize_capital(tf.constant(0.5))) == pytest.approx(0.0, abs=1e-6)
        assert float(norm.normalize_capital(tf.constant(3.0))) == pytest.approx(1.0, abs=1e-6)

    def test_denormalize_capital_roundtrip(self):
        config = _make_dl_config()
        norm = StateSpaceNormalizer(config)
        k = tf.constant([0.7, 1.5, 2.8])
        k_rt = norm.denormalize_capital(norm.normalize_capital(k))
        np.testing.assert_allclose(k_rt.numpy(), k.numpy(), atol=1e-5)

    def test_normalize_productivity_endpoints(self):
        config = _make_dl_config()
        norm = StateSpaceNormalizer(config)
        assert float(norm.normalize_productivity(tf.constant(0.5))) == pytest.approx(0.0, abs=1e-6)
        assert float(norm.normalize_productivity(tf.constant(1.5))) == pytest.approx(1.0, abs=1e-6)

    def test_normalize_debt_endpoints(self):
        config = _make_dl_config()
        norm = StateSpaceNormalizer(config)
        assert float(norm.normalize_debt(tf.constant(0.0))) == pytest.approx(0.0, abs=1e-6)
        assert float(norm.normalize_debt(tf.constant(2.0))) == pytest.approx(1.0, abs=1e-6)

    def test_denormalize_debt_roundtrip(self):
        config = _make_dl_config()
        norm = StateSpaceNormalizer(config)
        b = tf.constant([0.0, 0.5, 1.0, 2.0])
        b_rt = norm.denormalize_debt(norm.normalize_debt(b))
        np.testing.assert_allclose(b_rt.numpy(), b.numpy(), atol=1e-5)

    def test_missing_capital_raises(self):
        with pytest.raises(ValueError, match="Capital boundaries"):
            StateSpaceNormalizer(DeepLearningConfig(
                productivity_min=0.5, productivity_max=1.5
            ))

    def test_missing_productivity_raises(self):
        with pytest.raises(ValueError, match="Productivity boundaries"):
            StateSpaceNormalizer(DeepLearningConfig(
                capital_min=0.5, capital_max=3.0
            ))

    def test_batch_shape_preserved(self):
        config = _make_dl_config()
        norm = StateSpaceNormalizer(config)
        k = tf.random.uniform((16, 1), 0.5, 3.0)
        result = norm.normalize_capital(k)
        assert result.shape == (16, 1)

    def test_debt_defaults_when_unset(self):
        """Debt normalizer should not fail when debt bounds are None."""
        config = DeepLearningConfig(
            capital_min=0.5, capital_max=3.0,
            productivity_min=0.5, productivity_max=1.5,
        )
        norm = StateSpaceNormalizer(config)
        # Should use defaults (b_min=0, b_range=1)
        result = norm.normalize_debt(tf.constant(0.5))
        assert float(result) == pytest.approx(0.5, abs=1e-6)


class TestParamSpaceNormalizer:
    """Tests for parameter-space normalizer."""

    @pytest.fixture
    def bonds(self):
        return {
            'rho_min': 0.5, 'rho_max': 0.95,
            'std_min': 0.01, 'std_max': 0.2,
            'convex_min': 0.5, 'convex_max': 2.0,
            'fixed_min': 0.0, 'fixed_max': 0.1,
            'eta0_min': 0.0, 'eta0_max': 0.1,
            'eta1_min': 0.0, 'eta1_max': 0.1,
        }

    def test_rho_roundtrip(self, bonds):
        norm = ParamSpaceNormalizer(bonds)
        rho = tf.constant([0.6, 0.8])
        rho_rt = norm.denormalize_rho(norm.normalize_rho(rho))
        np.testing.assert_allclose(rho_rt.numpy(), rho.numpy(), atol=1e-5)

    def test_std_roundtrip(self, bonds):
        norm = ParamSpaceNormalizer(bonds)
        std = tf.constant([0.05, 0.15])
        std_rt = norm.denormalize_std(norm.normalize_std(std))
        np.testing.assert_allclose(std_rt.numpy(), std.numpy(), atol=1e-5)

    def test_endpoints_to_zero_one(self, bonds):
        norm = ParamSpaceNormalizer(bonds)
        assert float(norm.normalize_rho(tf.constant(0.5))) == pytest.approx(0.0, abs=1e-6)
        assert float(norm.normalize_rho(tf.constant(0.95))) == pytest.approx(1.0, abs=1e-6)

    def test_missing_rho_raises(self):
        with pytest.raises(ValueError, match="rho boundaries"):
            ParamSpaceNormalizer({
                'std_min': 0.01, 'std_max': 0.2,
                'convex_min': 0.5, 'convex_max': 2.0,
                'fixed_min': 0.0, 'fixed_max': 0.1,
            })
