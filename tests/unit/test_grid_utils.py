"""Unit tests for grid_utils: interpolation and grid utilities."""

from __future__ import annotations

import numpy as np
import tensorflow as tf

tf.config.set_visible_devices([], 'GPU')

import pytest

from econ_models.vfi.grids.grid_utils import (
    _interp_1d_batch_core,
    interp_1d_batch,
)


class TestInterp1dBatch:
    """Tests for 1-D batch linear interpolation."""

    def test_at_grid_points(self):
        """Interpolation at grid points returns exact values."""
        x = tf.constant([1.0, 2.0, 3.0, 4.0])
        y = tf.constant([10.0, 20.0, 30.0, 40.0])
        xq = tf.constant([1.0, 2.0, 3.0, 4.0])
        result = interp_1d_batch(x, y, xq).numpy()
        np.testing.assert_allclose(result, [10.0, 20.0, 30.0, 40.0], atol=1e-5)

    def test_midpoints(self):
        """Interpolation at midpoints returns mean of neighbors."""
        x = tf.constant([0.0, 1.0, 2.0])
        y = tf.constant([0.0, 10.0, 20.0])
        xq = tf.constant([0.5, 1.5])
        result = interp_1d_batch(x, y, xq).numpy()
        np.testing.assert_allclose(result, [5.0, 15.0], atol=1e-5)

    def test_extrapolation_clamps(self):
        """Out-of-bounds queries are clamped (no wild extrapolation)."""
        x = tf.constant([1.0, 2.0, 3.0])
        y = tf.constant([10.0, 20.0, 30.0])
        xq = tf.constant([0.0, 4.0])
        result = interp_1d_batch(x, y, xq).numpy()
        # Should clamp to boundary segment value
        assert result[0] == pytest.approx(10.0, abs=1e-3)
        assert result[1] == pytest.approx(30.0, abs=1e-3)

    def test_batch_dims(self):
        """Works with 2-D y_vals (batch dimension)."""
        x = tf.constant([0.0, 1.0, 2.0])
        # Shape (3, 4) — 4 batched series
        y = tf.constant([
            [0.0, 0.0, 0.0, 0.0],
            [10.0, 20.0, 30.0, 40.0],
            [20.0, 40.0, 60.0, 80.0],
        ])
        xq = tf.constant([0.5, 1.5])
        result = interp_1d_batch(x, y, xq).numpy()
        assert result.shape == (2, 4)
        np.testing.assert_allclose(result[0], [5.0, 10.0, 15.0, 20.0], atol=1e-5)
        np.testing.assert_allclose(result[1], [15.0, 30.0, 45.0, 60.0], atol=1e-5)

    def test_single_query(self):
        """Works with single query point."""
        x = tf.constant([0.0, 1.0])
        y = tf.constant([0.0, 10.0])
        xq = tf.constant([0.3])
        result = interp_1d_batch(x, y, xq).numpy()
        np.testing.assert_allclose(result, [3.0], atol=1e-5)

    def test_core_matches_compiled(self):
        """Core (undecorated) and compiled versions give same result."""
        x = tf.constant([0.0, 1.0, 2.0, 3.0])
        y = tf.constant([1.0, 4.0, 2.0, 7.0])
        xq = tf.constant([0.25, 1.75, 2.5])
        r1 = _interp_1d_batch_core(x, y, xq).numpy()
        r2 = interp_1d_batch(x, y, xq).numpy()
        np.testing.assert_allclose(r1, r2, atol=1e-6)
