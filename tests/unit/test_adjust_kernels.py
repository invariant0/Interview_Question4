"""Unit tests for adjust_kernels: adjust_tile_kernel.

Tests with synthetic tiles to verify shapes, dtypes, index bounds,
and constraint enforcement.
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf

tf.config.set_visible_devices([], 'GPU')

import pytest

from econ_models.vfi.kernels.adjust_kernels import (
    adjust_tile_kernel_core,
)
from econ_models.config.economic_params import EconomicParams


def _make_params():
    return EconomicParams(
        discount_factor=0.96,
        capital_share=0.3,
        depreciation_rate=0.1,
        productivity_persistence=0.9,
        productivity_std_dev=0.05,
        adjustment_cost_convex=1.0,
        adjustment_cost_fixed=0.0,
        equity_issuance_cost_fixed=0.0,
        equity_issuance_cost_linear=0.0,
        default_cost_proportional=0.3,
        corporate_tax_rate=0.2,
        risk_free_rate=0.04,
        collateral_recovery_fraction=0.5,
    )


class TestAdjustTileKernel:
    """Tests for adjust_tile_kernel_core."""

    def _make_inputs(self, ck=2, cb=3, ckp=2, cbp=3, nz=2):
        """Build synthetic inputs for the kernel."""
        flow_part_1 = tf.ones((ck, 1, ckp, 1, nz), dtype=tf.float32) * 5.0
        b_curr_chunk = tf.ones((cb,), dtype=tf.float32) * 0.1
        debt_comp = tf.ones((1, 1, ckp, cbp, nz), dtype=tf.float32) * 0.5
        cont_val = tf.ones((1, 1, ckp, cbp, nz), dtype=tf.float32) * 2.0
        coll_limit = tf.ones((1, 1, ckp, 1, 1), dtype=tf.float32) * 10.0
        b_next = tf.ones((1, 1, 1, cbp, 1), dtype=tf.float32) * 0.5
        penalty = tf.constant(-1e6, dtype=tf.float32)
        return (
            flow_part_1, b_curr_chunk, debt_comp, cont_val,
            coll_limit, b_next, penalty,
        )

    def test_output_shape(self):
        """Output shapes are (ck, cb, nz)."""
        ck, cb, ckp, cbp, nz = 2, 3, 2, 3, 2
        inputs = self._make_inputs(ck, cb, ckp, cbp, nz)
        params = _make_params()

        v_max, idx = adjust_tile_kernel_core(
            *inputs, enforce_constraint=False,
            ck=ck, cb=cb, ckp=ckp, cbp=cbp,
            n_productivity=nz, params=params,
        )
        assert v_max.shape == (ck, cb, nz)
        assert idx.shape == (ck, cb, nz)

    def test_output_dtype(self):
        """v_max is float32, idx is int32."""
        ck, cb, ckp, cbp, nz = 2, 2, 2, 2, 2
        inputs = self._make_inputs(ck, cb, ckp, cbp, nz)
        params = _make_params()

        v_max, idx = adjust_tile_kernel_core(
            *inputs, enforce_constraint=False,
            ck=ck, cb=cb, ckp=ckp, cbp=cbp,
            n_productivity=nz, params=params,
        )
        assert v_max.dtype == tf.float32
        assert idx.dtype == tf.int32

    def test_index_in_bounds(self):
        """Policy indices are in [0, ckp * cbp)."""
        ck, cb, ckp, cbp, nz = 3, 4, 5, 6, 2
        inputs = self._make_inputs(ck, cb, ckp, cbp, nz)
        params = _make_params()

        _, idx = adjust_tile_kernel_core(
            *inputs, enforce_constraint=False,
            ck=ck, cb=cb, ckp=ckp, cbp=cbp,
            n_productivity=nz, params=params,
        )
        assert np.all(idx.numpy() >= 0)
        assert np.all(idx.numpy() < ckp * cbp)

    def test_constraint_enforcement_penalty(self):
        """With constraint enforced and debt > limit, penalty is applied."""
        ck, cb, ckp, cbp, nz = 2, 2, 2, 2, 2
        params = _make_params()

        flow_part_1 = tf.ones((ck, 1, ckp, 1, nz), dtype=tf.float32)
        b_curr_chunk = tf.zeros((cb,), dtype=tf.float32)
        debt_comp = tf.ones((1, 1, ckp, cbp, nz), dtype=tf.float32)
        cont_val = tf.ones((1, 1, ckp, cbp, nz), dtype=tf.float32)
        # Set collateral limit very low
        coll_limit = tf.ones((1, 1, ckp, 1, 1), dtype=tf.float32) * 0.001
        # Set debt very high → violates constraint
        b_next = tf.ones((1, 1, 1, cbp, 1), dtype=tf.float32) * 100.0
        penalty = tf.constant(-1e6, dtype=tf.float32)

        v_max_nc, _ = adjust_tile_kernel_core(
            flow_part_1, b_curr_chunk, debt_comp, cont_val,
            coll_limit, b_next, penalty,
            enforce_constraint=False,
            ck=ck, cb=cb, ckp=ckp, cbp=cbp,
            n_productivity=nz, params=params,
        )
        v_max_c, _ = adjust_tile_kernel_core(
            flow_part_1, b_curr_chunk, debt_comp, cont_val,
            coll_limit, b_next, penalty,
            enforce_constraint=True,
            ck=ck, cb=cb, ckp=ckp, cbp=cbp,
            n_productivity=nz, params=params,
        )
        # With constraint, values should be much lower (penalty applied)
        assert np.all(v_max_c.numpy() < v_max_nc.numpy())

    def test_no_nan(self):
        """No NaN in output with normal inputs."""
        ck, cb, ckp, cbp, nz = 2, 2, 2, 2, 2
        inputs = self._make_inputs(ck, cb, ckp, cbp, nz)
        params = _make_params()

        v_max, idx = adjust_tile_kernel_core(
            *inputs, enforce_constraint=False,
            ck=ck, cb=cb, ckp=ckp, cbp=cbp,
            n_productivity=nz, params=params,
        )
        assert not np.any(np.isnan(v_max.numpy()))
