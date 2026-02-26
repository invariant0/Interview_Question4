"""Unit tests for target updater."""

import pytest
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

from econ_models.dl.components.target_updater import TargetUpdater


class TestTargetUpdater:
    """Tests for the Polyak-averaged target network sync."""

    def _make_pair(self):
        """Create a simple (source, target) network pair."""
        source = tf.keras.Sequential([
            tf.keras.layers.Dense(4, input_shape=(2,)),
            tf.keras.layers.Dense(1),
        ])
        target = tf.keras.Sequential([
            tf.keras.layers.Dense(4, input_shape=(2,)),
            tf.keras.layers.Dense(1),
        ])
        # Build both
        dummy = tf.zeros((1, 2))
        source(dummy)
        target(dummy)
        return source, target

    def test_decay_zero_makes_target_equal_source(self):
        source, target = self._make_pair()
        # Manually set different weights
        for tv in target.trainable_variables:
            tv.assign(tf.zeros_like(tv))
        updater = TargetUpdater([(source, target)])
        updater.update(tf.constant(0.0, dtype=tf.float32))

        for sv, tv in zip(source.trainable_variables, target.trainable_variables):
            tf.debugging.assert_near(sv, tv, atol=1e-6)

    def test_decay_one_leaves_target_unchanged(self):
        source, target = self._make_pair()
        original_weights = [tv.numpy().copy() for tv in target.trainable_variables]
        updater = TargetUpdater([(source, target)])
        updater.update(tf.constant(1.0, dtype=tf.float32))

        for orig, tv in zip(original_weights, target.trainable_variables):
            tf.debugging.assert_near(tf.constant(orig), tv, atol=1e-6)

    def test_decay_half_produces_average(self):
        source, target = self._make_pair()
        # Set known weights
        for sv, tv in zip(source.trainable_variables, target.trainable_variables):
            sv.assign(tf.ones_like(sv) * 2.0)
            tv.assign(tf.ones_like(tv) * 4.0)

        updater = TargetUpdater([(source, target)])
        updater.update(tf.constant(0.5, dtype=tf.float32))

        for tv in target.trainable_variables:
            # 0.5 * 4.0 + 0.5 * 2.0 = 3.0
            tf.debugging.assert_near(tv, tf.ones_like(tv) * 3.0, atol=1e-6)

    def test_build_compiled(self):
        source, target = self._make_pair()
        updater = TargetUpdater([(source, target)])
        compiled_fn = updater.build_compiled(use_xla=False)
        # Should be callable without error
        compiled_fn(tf.constant(0.99, dtype=tf.float32))
