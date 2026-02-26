"""Unit tests for tile_executor: execute_adjust_tiles.

Uses mock kernels to verify tiling coverage without GPU.
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf

tf.config.set_visible_devices([], 'GPU')

import pytest

from econ_models.vfi.chunking.tile_executor import execute_adjust_tiles


def _make_dummy_adjust_kernel(nz: int):
    """Return a mock kernel that returns fixed-value tiles."""

    def kernel(
        flow_part_1, b_curr_chunk, debt_comp, cont_val,
        coll_limit, b_next, penalty, enforce_constraint,
        ck, cb, ckp, cbp,
    ):
        # Return per-state max = 1.0 and idx = 0 for all cells
        v = tf.ones((ck, cb, nz), dtype=tf.float32)
        idx = tf.zeros((ck, cb, nz), dtype=tf.int32)
        return v, idx

    return kernel


def _make_dummy_chunk_accumulate():
    """Return a mock accumulator that takes the max."""

    def accumulate(v_chunk, idx_chunk, v_best, idx_best,
                   kp_start, bp_start, nb_t, cbp_t):
        improve = v_chunk > v_best
        v_out = tf.where(improve, v_chunk, v_best)
        idx_out = tf.where(improve, idx_chunk, idx_best)
        return v_out, idx_out

    return accumulate


class TestTileExecutor:
    """Tests for execute_adjust_tiles."""

    def test_output_shape(self):
        """Output has correct (nk, nb, nz) shape."""
        nk, nb, nz = 6, 8, 3
        v_adjust, policy = execute_adjust_tiles(
            n_capital=nk, n_debt=nb, n_productivity=nz,
            k_chunk_size=3, b_chunk_size=4,
            kp_chunk_size=3, bp_chunk_size=4,
            b_grid=tf.linspace(0.0, 1.0, nb),
            flow_part_1_full=tf.zeros((nk, 1, nk, 1, nz)),
            continuation_value=tf.zeros((nk, nb, nz)),
            debt_comp_full=tf.zeros((nk, nb, nz)),
            collateral_limit_5d=tf.zeros((1, 1, nk, 1, 1)),
            b_next_5d=tf.reshape(tf.linspace(0.0, 1.0, nb), (1, 1, 1, nb, 1)),
            penalty=tf.constant(-1e6),
            enforce_constraint=False,
            adjust_kernel=_make_dummy_adjust_kernel(nz),
            chunk_accumulate_fn=_make_dummy_chunk_accumulate(),
        )
        assert v_adjust.shape == (nk, nb, nz)
        assert policy.shape == (nk, nb, nz)

    def test_single_tile(self):
        """With chunk==full, only one tile is evaluated."""
        nk, nb, nz = 4, 4, 2

        call_count = [0]
        real_kernel = _make_dummy_adjust_kernel(nz)

        def counting_kernel(*args, **kwargs):
            call_count[0] += 1
            return real_kernel(*args, **kwargs)

        execute_adjust_tiles(
            n_capital=nk, n_debt=nb, n_productivity=nz,
            k_chunk_size=nk, b_chunk_size=nb,
            kp_chunk_size=nk, bp_chunk_size=nb,
            b_grid=tf.linspace(0.0, 1.0, nb),
            flow_part_1_full=tf.zeros((nk, 1, nk, 1, nz)),
            continuation_value=tf.zeros((nk, nb, nz)),
            debt_comp_full=tf.zeros((nk, nb, nz)),
            collateral_limit_5d=tf.zeros((1, 1, nk, 1, 1)),
            b_next_5d=tf.reshape(tf.linspace(0.0, 1.0, nb), (1, 1, 1, nb, 1)),
            penalty=tf.constant(-1e6),
            enforce_constraint=False,
            adjust_kernel=counting_kernel,
            chunk_accumulate_fn=_make_dummy_chunk_accumulate(),
        )
        assert call_count[0] == 1  # single tile

    def test_multi_tile_count(self):
        """Correct number of kernel calls for multi-tile case."""
        nk, nb, nz = 6, 8, 2
        k_cs, b_cs, kp_cs, bp_cs = 3, 4, 2, 4

        call_count = [0]
        real_kernel = _make_dummy_adjust_kernel(nz)

        def counting_kernel(*args, **kwargs):
            call_count[0] += 1
            return real_kernel(*args, **kwargs)

        execute_adjust_tiles(
            n_capital=nk, n_debt=nb, n_productivity=nz,
            k_chunk_size=k_cs, b_chunk_size=b_cs,
            kp_chunk_size=kp_cs, bp_chunk_size=bp_cs,
            b_grid=tf.linspace(0.0, 1.0, nb),
            flow_part_1_full=tf.zeros((nk, 1, nk, 1, nz)),
            continuation_value=tf.zeros((nk, nb, nz)),
            debt_comp_full=tf.zeros((nk, nb, nz)),
            collateral_limit_5d=tf.zeros((1, 1, nk, 1, 1)),
            b_next_5d=tf.reshape(tf.linspace(0.0, 1.0, nb), (1, 1, 1, nb, 1)),
            penalty=tf.constant(-1e6),
            enforce_constraint=False,
            adjust_kernel=counting_kernel,
            chunk_accumulate_fn=_make_dummy_chunk_accumulate(),
        )

        import math
        n_k_tiles = math.ceil(nk / k_cs)
        n_b_tiles = math.ceil(nb / b_cs)
        n_kp_tiles = math.ceil(nk / kp_cs)
        n_bp_tiles = math.ceil(nb / bp_cs)
        expected = n_k_tiles * n_b_tiles * n_kp_tiles * n_bp_tiles
        assert call_count[0] == expected

    def test_all_values_populated(self):
        """No -inf values remain after tiling."""
        nk, nb, nz = 5, 5, 2
        v_adjust, _ = execute_adjust_tiles(
            n_capital=nk, n_debt=nb, n_productivity=nz,
            k_chunk_size=2, b_chunk_size=2,
            kp_chunk_size=2, bp_chunk_size=2,
            b_grid=tf.linspace(0.0, 1.0, nb),
            flow_part_1_full=tf.zeros((nk, 1, nk, 1, nz)),
            continuation_value=tf.zeros((nk, nb, nz)),
            debt_comp_full=tf.zeros((nk, nb, nz)),
            collateral_limit_5d=tf.zeros((1, 1, nk, 1, 1)),
            b_next_5d=tf.reshape(tf.linspace(0.0, 1.0, nb), (1, 1, 1, nb, 1)),
            penalty=tf.constant(-1e6),
            enforce_constraint=False,
            adjust_kernel=_make_dummy_adjust_kernel(nz),
            chunk_accumulate_fn=_make_dummy_chunk_accumulate(),
        )
        assert not np.any(np.isinf(v_adjust.numpy()))
