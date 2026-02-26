"""Unit tests for wait_kernels: compute_flow_wait_xla, wait_branch_reduce."""

from __future__ import annotations

import numpy as np
import tensorflow as tf

tf.config.set_visible_devices([], 'GPU')

import pytest

from econ_models.vfi.kernels.wait_kernels import (
    compute_flow_wait_xla_core,
    wait_branch_reduce_core,
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


class TestComputeFlowWaitXla:
    """Tests for compute_flow_wait_xla_core."""

    def test_output_shape(self):
        """Flow tensor has (nk, nb, nb, nz) shape."""
        nk, nb, nz = 3, 4, 2
        k_grid = tf.linspace(0.5, 3.0, nk)
        b_grid = tf.linspace(0.0, 1.0, nb)
        z_grid = tf.linspace(-0.1, 0.1, nz)
        k_dep = k_grid * (1.0 - 0.1)
        q_wait = tf.ones((nk, nb, nz))
        params = _make_params()

        flow = compute_flow_wait_xla_core(
            k_grid, b_grid, z_grid, k_dep, q_wait,
            nk, nb, nz,
            z_min=float(z_grid[0]),
            params=params,
            collateral_violation_penalty=-1e6,
            collateral_recovery_fraction=0.5,
            enforce_constraint=False,
        )
        assert flow.shape == (nk, nb, nb, nz)

    def test_no_nan(self):
        """No NaN in output with normal inputs."""
        nk, nb, nz = 3, 4, 2
        k_grid = tf.linspace(0.5, 3.0, nk)
        b_grid = tf.linspace(0.0, 1.0, nb)
        z_grid = tf.linspace(-0.1, 0.1, nz)
        k_dep = k_grid * 0.9
        q_wait = tf.ones((nk, nb, nz)) * 0.96
        params = _make_params()

        flow = compute_flow_wait_xla_core(
            k_grid, b_grid, z_grid, k_dep, q_wait,
            nk, nb, nz,
            z_min=float(z_grid[0]),
            params=params,
            collateral_violation_penalty=-1e6,
            collateral_recovery_fraction=0.5,
            enforce_constraint=False,
        )
        assert not np.any(np.isnan(flow.numpy()))

    def test_constraint_lowers_values(self):
        """Enforcing constraint with high debt → penalised cells."""
        nk, nb, nz = 3, 4, 2
        k_grid = tf.linspace(0.5, 3.0, nk)
        b_grid = tf.linspace(0.0, 50.0, nb)  # high debt
        z_grid = tf.linspace(-0.1, 0.1, nz)
        k_dep = k_grid * 0.9
        q_wait = tf.ones((nk, nb, nz)) * 0.96
        params = _make_params()

        flow_nc = compute_flow_wait_xla_core(
            k_grid, b_grid, z_grid, k_dep, q_wait,
            nk, nb, nz,
            z_min=float(z_grid[0]),
            params=params,
            collateral_violation_penalty=-1e6,
            collateral_recovery_fraction=0.5,
            enforce_constraint=False,
        )
        flow_c = compute_flow_wait_xla_core(
            k_grid, b_grid, z_grid, k_dep, q_wait,
            nk, nb, nz,
            z_min=float(z_grid[0]),
            params=params,
            collateral_violation_penalty=-1e6,
            collateral_recovery_fraction=0.5,
            enforce_constraint=True,
        )
        # Constrained flow should have some very negative values
        assert np.min(flow_c.numpy()) < np.min(flow_nc.numpy())


class TestWaitBranchReduce:
    """Tests for wait_branch_reduce_core."""

    def test_output_shape(self):
        """Output shapes are (nk, nb, nz)."""
        nk, nb, nz = 4, 5, 3
        k_grid = tf.linspace(0.5, 3.0, nk)
        k_dep = k_grid * 0.9
        flow_wait = tf.ones((nk, nb, nb, nz))
        beta_ev = tf.ones((nk, nb, nz))

        v_wait, idx = wait_branch_reduce_core(
            k_grid, k_dep, flow_wait, beta_ev,
            nk, nb, nz,
        )
        assert v_wait.shape == (nk, nb, nz)
        assert idx.shape == (nk, nb, nz)

    def test_index_bounds(self):
        """Optimal debt indices are in [0, nb)."""
        nk, nb, nz = 3, 6, 2
        k_grid = tf.linspace(0.5, 3.0, nk)
        k_dep = k_grid * 0.9
        flow_wait = tf.random.uniform((nk, nb, nb, nz))
        beta_ev = tf.random.uniform((nk, nb, nz))

        _, idx = wait_branch_reduce_core(
            k_grid, k_dep, flow_wait, beta_ev,
            nk, nb, nz,
        )
        assert np.all(idx.numpy() >= 0)
        assert np.all(idx.numpy() < nb)

    def test_known_optimal_debt(self):
        """When one debt index dominates, policy picks it."""
        nk, nb, nz = 2, 4, 2
        k_grid = tf.linspace(0.5, 2.0, nk)
        k_dep = k_grid * 0.9

        # Make debt index 2 clearly dominant
        flow_wait = tf.zeros((nk, nb, nb, nz))
        bonus = tf.zeros((nk, nb, nb, nz))
        # flow_wait[:, :, 2, :] = 100
        indices = []
        for ki in range(nk):
            for bi in range(nb):
                for zi in range(nz):
                    indices.append([ki, bi, 2, zi])
        indices = tf.constant(indices)
        updates = tf.ones((len(indices),)) * 100.0
        flow_wait = tf.tensor_scatter_nd_update(flow_wait, indices, updates)

        beta_ev = tf.zeros((nk, nb, nz))

        _, idx = wait_branch_reduce_core(
            k_grid, k_dep, flow_wait, beta_ev,
            nk, nb, nz,
        )
        # All should pick debt index 2
        np.testing.assert_array_equal(idx.numpy(), 2)
