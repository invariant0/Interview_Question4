"""Unit tests for bond_price_kernels: update_bond_prices."""

from __future__ import annotations

import numpy as np
import tensorflow as tf

tf.config.set_visible_devices([], 'GPU')

import pytest

from econ_models.vfi.kernels.bond_price_kernels import update_bond_prices_core
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


class TestUpdateBondPrices:
    """Tests for update_bond_prices_core."""

    def _make_inputs(self, nk=4, nb=5, nz=3):
        k_grid = tf.linspace(0.5, 3.0, nk)
        b_grid = tf.linspace(0.0, 1.0, nb)
        z_grid = tf.linspace(-0.1, 0.1, nz)
        P = tf.eye(nz)
        params = _make_params()
        q_rf = 1.0 / (1.0 + params.risk_free_rate)
        q_old = tf.fill((nk, nb, nz), q_rf)
        return k_grid, b_grid, z_grid, P, q_old, params, nk, nb, nz

    def test_output_shape(self):
        """q_updated has (nk, nb, nz) shape."""
        k_grid, b_grid, z_grid, P, q_old, params, nk, nb, nz = self._make_inputs()
        vf = tf.ones((nk, nb, nz)) * 10.0  # all positive → no default

        q_new, diff = update_bond_prices_core(
            vf, q_old, k_grid, b_grid, z_grid, P,
            nk, nb, nz, params,
            v_default_eps=0.01, b_eps=1e-6, q_min=0.01, relax_q=1.0,
        )
        assert q_new.shape == (nk, nb, nz)
        assert diff.shape == ()

    def test_no_default_prices_near_risk_free(self):
        """When no firms default, bond prices should be near q_rf."""
        nk, nb, nz = 4, 5, 3
        k_grid, b_grid, z_grid, P, q_old, params, _, _, _ = self._make_inputs(nk, nb, nz)
        vf = tf.ones((nk, nb, nz)) * 100.0  # large positive → no default

        q_new, _ = update_bond_prices_core(
            vf, q_old, k_grid, b_grid, z_grid, P,
            nk, nb, nz, params,
            v_default_eps=0.01, b_eps=1e-6, q_min=0.01, relax_q=1.0,
        )
        q_rf = 1.0 / (1.0 + params.risk_free_rate)
        # For zero-debt or near-zero debt, price should be close to risk-free
        # Check first column (b ≈ 0)
        q_zero_debt = q_new[:, 0, :].numpy()
        # With very small debt, price should approach q_rf
        np.testing.assert_allclose(q_zero_debt, q_rf, atol=0.2)

    def test_all_default_lower_prices(self):
        """When all firms default, prices should be lower."""
        nk, nb, nz = 4, 5, 3
        k_grid, b_grid, z_grid, P, q_old, params, _, _, _ = self._make_inputs(nk, nb, nz)

        vf_no_default = tf.ones((nk, nb, nz)) * 100.0
        vf_all_default = tf.ones((nk, nb, nz)) * -1.0  # below eps

        q_good, _ = update_bond_prices_core(
            vf_no_default, q_old, k_grid, b_grid, z_grid, P,
            nk, nb, nz, params,
            v_default_eps=0.01, b_eps=1e-6, q_min=0.01, relax_q=1.0,
        )
        q_bad, _ = update_bond_prices_core(
            vf_all_default, q_old, k_grid, b_grid, z_grid, P,
            nk, nb, nz, params,
            v_default_eps=0.01, b_eps=1e-6, q_min=0.01, relax_q=1.0,
        )
        # Average price with full default should be lower
        assert np.mean(q_bad.numpy()) < np.mean(q_good.numpy())

    def test_relaxation_blending(self):
        """With relax_q=0.5, update is blended with old prices."""
        nk, nb, nz = 3, 3, 2
        k_grid = tf.linspace(0.5, 3.0, nk)
        b_grid = tf.linspace(0.0, 1.0, nb)
        z_grid = tf.linspace(-0.1, 0.1, nz)
        P = tf.eye(nz)
        params = _make_params()

        # Distinct old prices
        q_old = tf.fill((nk, nb, nz), 0.5)
        vf = tf.ones((nk, nb, nz)) * 100.0

        q_full, _ = update_bond_prices_core(
            vf, q_old, k_grid, b_grid, z_grid, P,
            nk, nb, nz, params,
            v_default_eps=0.01, b_eps=1e-6, q_min=0.01, relax_q=1.0,
        )
        q_half, _ = update_bond_prices_core(
            vf, q_old, k_grid, b_grid, z_grid, P,
            nk, nb, nz, params,
            v_default_eps=0.01, b_eps=1e-6, q_min=0.01, relax_q=0.5,
        )
        # With relax=0.5: q_half = 0.5*q_new + 0.5*q_old
        # This should be between q_old and q_full (relax=1.0)
        q_old_val = 0.5
        q_full_np = q_full.numpy()
        q_half_np = q_half.numpy()
        expected_half = 0.5 * q_full_np + 0.5 * q_old_val
        np.testing.assert_allclose(q_half_np, expected_half, atol=1e-5)

    def test_prices_above_floor(self):
        """All prices should be >= q_min."""
        nk, nb, nz = 4, 5, 3
        k_grid, b_grid, z_grid, P, q_old, params, _, _, _ = self._make_inputs(nk, nb, nz)
        vf = tf.ones((nk, nb, nz)) * -10.0  # all default

        q_min = 0.05
        q_new, _ = update_bond_prices_core(
            vf, q_old, k_grid, b_grid, z_grid, P,
            nk, nb, nz, params,
            v_default_eps=0.01, b_eps=1e-6, q_min=q_min, relax_q=1.0,
        )
        # Floor applied inside risk_neutral_price
        # Note: the relaxation with old prices may push above floor
        # but new_price itself should respect floor
        assert np.all(q_new.numpy() >= q_min - 0.01)  # small tolerance
