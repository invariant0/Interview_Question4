"""Unit tests for NeuralNetFactory (core/nets.py)."""

from __future__ import annotations

import numpy as np
import tensorflow as tf

tf.config.set_visible_devices([], 'GPU')

import pytest

from econ_models.core.nets import NeuralNetFactory
from econ_models.config.dl_config import DeepLearningConfig


def _make_config(**overrides):
    defaults = dict(
        hidden_layers=(32, 32),
        activation_function='relu',
        use_layer_norm=False,
        learning_rate=1e-3,
    )
    defaults.update(overrides)
    return DeepLearningConfig(**defaults)


class TestBuildMLP:
    """Tests for build_mlp."""

    def test_output_shape_1d(self):
        config = _make_config()
        net = NeuralNetFactory.build_mlp(input_dim=2, output_dim=1, config=config)
        out = net(tf.zeros((4, 2)))
        assert out.shape == (4, 1)

    def test_output_shape_multi(self):
        config = _make_config()
        net = NeuralNetFactory.build_mlp(input_dim=3, output_dim=5, config=config)
        out = net(tf.zeros((8, 3)))
        assert out.shape == (8, 5)

    def test_sigmoid_output_range(self):
        config = _make_config()
        net = NeuralNetFactory.build_mlp(
            input_dim=2, output_dim=1, config=config,
            output_activation='sigmoid',
        )
        out = net(tf.random.uniform((100, 2), -5, 5))
        assert float(tf.reduce_min(out)) >= 0.0
        assert float(tf.reduce_max(out)) <= 1.0

    def test_linear_output_unbounded(self):
        config = _make_config()
        net = NeuralNetFactory.build_mlp(
            input_dim=2, output_dim=1, config=config,
            output_activation='linear',
        )
        out = net(tf.random.normal((100, 2)))
        # Linear activation should allow negative values
        assert float(tf.reduce_min(out)) < 0 or float(tf.reduce_max(out)) > 0

    def test_scale_factor(self):
        config = _make_config()
        net_unscaled = NeuralNetFactory.build_mlp(
            input_dim=2, output_dim=1, config=config, name="unscaled",
        )
        net_scaled = NeuralNetFactory.build_mlp(
            input_dim=2, output_dim=1, config=config,
            scale_factor=10.0, name="scaled",
        )
        # Copy weights from unscaled to scaled
        x = tf.constant([[1.0, 0.5]])
        net_unscaled(x)  # build
        net_scaled(x)   # build
        for sv, tv in zip(net_unscaled.trainable_variables, net_scaled.trainable_variables):
            tv.assign(sv)
        out_u = net_unscaled(x)
        out_s = net_scaled(x)
        np.testing.assert_allclose(out_s.numpy(), out_u.numpy() * 10.0, atol=1e-5)

    def test_layer_norm_builds(self):
        config = _make_config(use_layer_norm=True)
        net = NeuralNetFactory.build_mlp(input_dim=2, output_dim=1, config=config)
        out = net(tf.zeros((4, 2)))
        assert out.shape == (4, 1)
        # Layer norm should add LayerNormalization layers
        layer_names = [l.name for l in net.layers]
        assert any('LayerNorm' in n for n in layer_names)

    def test_custom_hidden_activation(self):
        config = _make_config(activation_function='tanh')
        net = NeuralNetFactory.build_mlp(
            input_dim=2, output_dim=1, config=config,
            hidden_activation='elu',  # override
        )
        out = net(tf.random.normal((4, 2)))
        assert out.shape == (4, 1)

    def test_model_is_trainable(self):
        config = _make_config()
        net = NeuralNetFactory.build_mlp(input_dim=2, output_dim=1, config=config)
        x = tf.constant([[1.0, 0.5]])
        with tf.GradientTape() as tape:
            y = net(x, training=True)
            loss = tf.reduce_sum(y)
        grads = tape.gradient(loss, net.trainable_variables)
        assert all(g is not None for g in grads)


class TestSoftUpdate:
    """Tests for Polyak-averaged soft_update."""

    def _make_pair(self, input_dim=2):
        config = _make_config()
        source = NeuralNetFactory.build_mlp(input_dim=input_dim, output_dim=1,
                                             config=config, name="source")
        target = NeuralNetFactory.build_mlp(input_dim=input_dim, output_dim=1,
                                             config=config, name="target")
        dummy = tf.zeros((1, input_dim))
        source(dummy)
        target(dummy)
        return source, target

    def test_polyak_zero_keeps_target(self):
        """polyak=0 → 0*src + 1*tgt → target unchanged."""
        source, target = self._make_pair()
        original = [tv.numpy().copy() for tv in target.trainable_variables]
        NeuralNetFactory.soft_update(source, target, polyak=0.0)
        for orig, tv in zip(original, target.trainable_variables):
            np.testing.assert_allclose(tv.numpy(), orig, atol=1e-6)

    def test_polyak_one_copies_source(self):
        """polyak=1 → 1*src + 0*tgt → target becomes source."""
        source, target = self._make_pair()
        NeuralNetFactory.soft_update(source, target, polyak=1.0)
        for sv, tv in zip(source.trainable_variables, target.trainable_variables):
            np.testing.assert_allclose(tv.numpy(), sv.numpy(), atol=1e-6)

    def test_polyak_half_averages(self):
        source, target = self._make_pair()
        for sv in source.trainable_variables:
            sv.assign(tf.ones_like(sv) * 2.0)
        for tv in target.trainable_variables:
            tv.assign(tf.ones_like(tv) * 4.0)
        NeuralNetFactory.soft_update(source, target, polyak=0.5)
        for tv in target.trainable_variables:
            np.testing.assert_allclose(tv.numpy(), 3.0, atol=1e-6)


class TestBuildMultiHeadMLP:
    """Tests for build_multi_head_mlp."""

    def test_two_heads(self):
        config = _make_config()
        net = NeuralNetFactory.build_multi_head_mlp(
            input_dim=3,
            head_configs=[
                {"name": "capital", "output_dim": 1, "activation": "sigmoid"},
                {"name": "debt", "output_dim": 1, "activation": "sigmoid"},
            ],
            config=config,
        )
        out = net(tf.zeros((4, 3)))
        assert len(out) == 2
        assert out[0].shape == (4, 1)
        assert out[1].shape == (4, 1)

    def test_sigmoid_heads_range(self):
        config = _make_config()
        net = NeuralNetFactory.build_multi_head_mlp(
            input_dim=3,
            head_configs=[
                {"name": "a", "output_dim": 1, "activation": "sigmoid"},
            ],
            config=config,
        )
        out = net(tf.random.normal((100, 3)))
        assert float(tf.reduce_min(out[0])) >= 0.0
        assert float(tf.reduce_max(out[0])) <= 1.0
