"""Unit tests for chunk_accumulate kernel.

Tests tile-index remapping and running-best accumulation logic.
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf

tf.config.set_visible_devices([], 'GPU')

import pytest

from econ_models.vfi.kernels.chunk_accumulate import chunk_accumulate


class TestChunkAccumulate:
    """Tests for chunk_accumulate."""

    def _t(self, val, dtype=tf.int32):
        return tf.constant(val, dtype=dtype)

    def test_single_tile_global_index(self):
        """When offsets are 0 the global index equals local index."""
        ck, cb, nz = 2, 3, 2
        nbp = 4

        v_chunk = tf.constant(np.random.rand(ck, cb, nz).astype(np.float32))
        idx_chunk = tf.constant(np.random.randint(0, nbp, (ck, cb, nz)), dtype=tf.int32)

        neg_inf = tf.fill((ck, cb, nz), float('-inf'))
        idx_best = tf.zeros((ck, cb, nz), dtype=tf.int32)

        v_best, idx_best_out = chunk_accumulate(
            v_chunk, idx_chunk,
            neg_inf, idx_best,
            self._t(0), self._t(0), self._t(nbp), self._t(nbp),
        )

        np.testing.assert_allclose(v_best.numpy(), v_chunk.numpy())
        np.testing.assert_array_equal(idx_best_out.numpy(), idx_chunk.numpy())

    def test_two_sequential_tiles(self):
        """Second tile only overwrites where it improves."""
        ck, cb, nz = 2, 2, 2
        nb = 6
        cbp = 3

        # Tile 1: kp_start=0, bp_start=0
        v1 = tf.constant([[[5.0, 3.0], [2.0, 8.0]],
                          [[1.0, 4.0], [7.0, 2.0]]])
        idx1 = tf.constant([[[0, 1], [2, 0]],
                            [[1, 2], [0, 1]]], dtype=tf.int32)
        neg_inf = tf.fill((ck, cb, nz), float('-inf'))
        idx0 = tf.zeros((ck, cb, nz), dtype=tf.int32)

        v_best, idx_best = chunk_accumulate(
            v1, idx1, neg_inf, idx0,
            self._t(0), self._t(0), self._t(nb), self._t(cbp),
        )

        # Tile 2: kp_start=0, bp_start=3 — only improves some cells
        v2 = tf.constant([[[6.0, 1.0], [1.0, 9.0]],
                          [[0.0, 5.0], [7.5, 1.0]]])
        idx2 = tf.constant([[[0, 0], [1, 2]],
                            [[0, 1], [2, 0]]], dtype=tf.int32)

        v_best2, idx_best2 = chunk_accumulate(
            v2, idx2, v_best, idx_best,
            self._t(0), self._t(3), self._t(nb), self._t(cbp),
        )

        # Cell [0,0,0]: tile2 (6.0) > tile1 (5.0) → should update
        assert float(v_best2[0, 0, 0]) == pytest.approx(6.0)
        # Cell [0,0,1]: tile2 (1.0) < tile1 (3.0) → no update
        assert float(v_best2[0, 0, 1]) == pytest.approx(3.0)

    def test_no_improvement(self):
        """If new tile is all lower, running-best unchanged."""
        ck, cb, nz = 2, 2, 2
        nb, cbp = 4, 4

        v_best = tf.fill((ck, cb, nz), 100.0)
        idx_best = tf.zeros((ck, cb, nz), dtype=tf.int32)

        v_chunk = tf.fill((ck, cb, nz), 50.0)
        idx_chunk = tf.ones((ck, cb, nz), dtype=tf.int32)

        v_out, idx_out = chunk_accumulate(
            v_chunk, idx_chunk, v_best, idx_best,
            self._t(0), self._t(0), self._t(nb), self._t(cbp),
        )

        np.testing.assert_allclose(v_out.numpy(), v_best.numpy())
        np.testing.assert_array_equal(idx_out.numpy(), idx_best.numpy())

    def test_global_index_with_offset(self):
        """Verify global flat-index arithmetic with non-zero offsets."""
        ck, cb, nz = 1, 1, 1
        nb_total = 10
        cbp = 3
        kp_start = 2
        bp_start = 4

        # Local idx: kp_local=1, bp_local=2
        idx_chunk = tf.constant([[[1 * 3 + 2]]], dtype=tf.int32)  # = 5
        v_chunk = tf.constant([[[10.0]]])
        neg_inf = tf.fill((1, 1, 1), float('-inf'))
        idx0 = tf.zeros((1, 1, 1), dtype=tf.int32)

        _, idx_out = chunk_accumulate(
            v_chunk, idx_chunk, neg_inf, idx0,
            self._t(kp_start), self._t(bp_start),
            self._t(nb_total), self._t(cbp),
        )

        # Global flat: (1 + 2) * 10 + (2 + 4) = 36
        expected_global = (1 + kp_start) * nb_total + (2 + bp_start)
        assert int(idx_out[0, 0, 0]) == expected_global
