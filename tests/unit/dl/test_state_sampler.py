"""Unit tests for StateSampler."""

from __future__ import annotations

import tensorflow as tf

tf.config.set_visible_devices([], 'GPU')

import pytest

from econ_models.core.sampling.state_sampler import StateSampler


@pytest.fixture
def bonds_config():
    return dict(
        k_min=0.5, k_max=5.0,
        z_min=0.3, z_max=2.0,
        b_min=0.0, b_max=4.0,
    )


class TestSampleStates:
    """Tests for StateSampler.sample_states (CPU / Beta-based)."""

    def test_returns_two_tensors_without_debt(self, bonds_config):
        result = StateSampler.sample_states(32, bonds_config, include_debt=False)
        assert len(result) == 2
        k, z = result
        assert k.shape == (32, 1)
        assert z.shape == (32, 1)

    def test_returns_three_tensors_with_debt(self, bonds_config):
        result = StateSampler.sample_states(32, bonds_config, include_debt=True)
        assert len(result) == 3
        k, b, z = result
        assert k.shape == (32, 1)
        assert b.shape == (32, 1)
        assert z.shape == (32, 1)

    def test_capital_within_bounds(self, bonds_config):
        k, _ = StateSampler.sample_states(200, bonds_config)
        assert float(tf.reduce_min(k)) >= bonds_config['k_min'] - 1e-6
        assert float(tf.reduce_max(k)) <= bonds_config['k_max'] + 1e-6

    def test_productivity_within_bounds(self, bonds_config):
        _, z = StateSampler.sample_states(200, bonds_config)
        assert float(tf.reduce_min(z)) >= bonds_config['z_min'] - 1e-6
        assert float(tf.reduce_max(z)) <= bonds_config['z_max'] + 1e-6

    def test_debt_within_bounds(self, bonds_config):
        _, b, _ = StateSampler.sample_states(200, bonds_config, include_debt=True)
        assert float(tf.reduce_min(b)) >= bonds_config['b_min'] - 1e-6
        assert float(tf.reduce_max(b)) <= bonds_config['b_max'] + 1e-6


class TestSampleStatesGPU:
    """Tests for StateSampler.sample_states_gpu (uniform / XLA-safe)."""

    def test_returns_two_tensors_without_debt(self, bonds_config):
        result = StateSampler.sample_states_gpu(32, bonds_config, include_debt=False)
        assert len(result) == 2
        k, z = result
        assert k.shape == (32, 1)
        assert z.shape == (32, 1)

    def test_returns_three_tensors_with_debt(self, bonds_config):
        result = StateSampler.sample_states_gpu(32, bonds_config, include_debt=True)
        assert len(result) == 3

    def test_capital_bounds_gpu(self, bonds_config):
        k, _ = StateSampler.sample_states_gpu(200, bonds_config)
        assert float(tf.reduce_min(k)) >= bonds_config['k_min'] - 1e-6
        assert float(tf.reduce_max(k)) <= bonds_config['k_max'] + 1e-6

    def test_productivity_bounds_gpu(self, bonds_config):
        _, z = StateSampler.sample_states_gpu(200, bonds_config)
        assert float(tf.reduce_min(z)) >= bonds_config['z_min'] - 1e-6
        assert float(tf.reduce_max(z)) <= bonds_config['z_max'] + 1e-6

    def test_debt_bounds_gpu(self, bonds_config):
        _, b, _ = StateSampler.sample_states_gpu(200, bonds_config, include_debt=True)
        assert float(tf.reduce_min(b)) >= bonds_config['b_min'] - 1e-6
        assert float(tf.reduce_max(b)) <= bonds_config['b_max'] + 1e-6
