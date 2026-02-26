"""Unit tests for shock sampler."""

import pytest
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

from econ_models.dl.components.shock_sampler import ShockSampler


class TestShockSampler:
    """Tests for ShockSampler."""

    def test_output_shape(self):
        sampler = ShockSampler(productivity_std_dev=0.05, is_distributional=False)
        result = sampler.sample(batch_size=16, n_draws=1)
        assert result.shape == (16, 1)

    def test_output_shape_multi_draw(self):
        sampler = ShockSampler(productivity_std_dev=0.05, is_distributional=False)
        result = sampler.sample(batch_size=8, n_draws=10)
        assert result.shape == (8, 10)

    def test_mean_approximately_zero(self):
        sampler = ShockSampler(productivity_std_dev=0.05, is_distributional=False)
        tf.random.set_seed(42)
        result = sampler.sample(batch_size=10000, n_draws=1)
        assert abs(float(tf.reduce_mean(result))) < 0.01

    def test_std_matches_non_distributional(self):
        std_dev = 0.05
        sampler = ShockSampler(productivity_std_dev=std_dev, is_distributional=False)
        tf.random.set_seed(42)
        result = sampler.sample(batch_size=50000, n_draws=1)
        empirical_std = float(tf.math.reduce_std(result))
        assert abs(empirical_std - std_dev) < 0.005

    def test_distributional_gives_unit_variance(self):
        sampler = ShockSampler(productivity_std_dev=0.05, is_distributional=True)
        tf.random.set_seed(42)
        result = sampler.sample(batch_size=50000, n_draws=1)
        empirical_std = float(tf.math.reduce_std(result))
        assert abs(empirical_std - 1.0) < 0.05

    def test_sample_for_bond_pricing_shape(self):
        sampler = ShockSampler(productivity_std_dev=0.05, is_distributional=False)
        result = sampler.sample_for_bond_pricing(batch_size=8, n_samples=20)
        assert result.shape == (8, 20)

    def test_sample_standard_unit_variance(self):
        sampler = ShockSampler(productivity_std_dev=0.05, is_distributional=False)
        tf.random.set_seed(42)
        result = sampler.sample_standard(batch_size=50000, n_draws=1)
        empirical_std = float(tf.math.reduce_std(result))
        assert abs(empirical_std - 1.0) < 0.05
