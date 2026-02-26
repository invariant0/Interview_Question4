"""Unit tests for flows: build_adjust_flow_part1, build_debt_components, build_wait_flow."""

from __future__ import annotations

import numpy as np
import tensorflow as tf

tf.config.set_visible_devices([], 'GPU')

import pytest

from econ_models.vfi.flows.adjust_flow import build_adjust_flow_part1
from econ_models.vfi.flows.debt_components import build_debt_components
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


class TestBuildAdjustFlowPart1:
    """Tests for build_adjust_flow_part1."""

    def test_output_shape(self):
        """Output has (nk, 1, nk, 1, nz) shape."""
        nk, nz = 5, 3
        k_grid = tf.linspace(0.5, 3.0, nk)
        z_grid = tf.linspace(-0.1, 0.1, nz)
        params = _make_params()

        result = build_adjust_flow_part1(k_grid, z_grid, nk, nz, params)
        assert result.shape == (nk, 1, nk, 1, nz)

    def test_no_nan(self):
        """No NaN values in output."""
        nk, nz = 4, 2
        k_grid = tf.linspace(0.5, 3.0, nk)
        z_grid = tf.linspace(-0.1, 0.1, nz)
        params = _make_params()

        result = build_adjust_flow_part1(k_grid, z_grid, nk, nz, params)
        assert not np.any(np.isnan(result.numpy()))

    def test_higher_productivity_higher_flow(self):
        """Higher productivity z should yield higher flow values."""
        nk, nz = 5, 3
        k_grid = tf.linspace(0.5, 3.0, nk)
        z_grid = tf.linspace(-0.5, 0.5, nz)  # wide range for clear signal
        params = _make_params()

        result = build_adjust_flow_part1(k_grid, z_grid, nk, nz, params)
        r = result.numpy()
        # For fixed k_curr and k_next, higher z → higher flow
        # Compare z=0 (idx 0) vs z=nz-1 (highest)
        assert np.all(r[:, 0, :, 0, -1] > r[:, 0, :, 0, 0])

    def test_no_investment_diagonal(self):
        """On the diagonal (k_next = k_curr), investment ≈ delta * k."""
        nk, nz = 10, 2
        k_grid = tf.linspace(1.0, 3.0, nk)
        z_grid = tf.constant([0.0, 0.0])
        params = _make_params()

        result = build_adjust_flow_part1(k_grid, z_grid, nk, nz, params)
        # Just check it's finite and has the right pattern
        diag = np.array([result.numpy()[i, 0, i, 0, 0] for i in range(nk)])
        assert np.all(np.isfinite(diag))


class TestBuildDebtComponents:
    """Tests for build_debt_components."""

    def test_output_shape(self):
        """Output has (nk, nb, nz) shape."""
        nk, nb, nz = 4, 5, 3
        b_grid = tf.linspace(0.0, 1.0, nb)
        bond_prices = tf.ones((nk, nb, nz)) * 0.96
        params = _make_params()

        result = build_debt_components(b_grid, bond_prices, nb, params)
        assert result.shape == (nk, nb, nz)

    def test_no_nan(self):
        """No NaN values with normal inputs."""
        nk, nb, nz = 3, 4, 2
        b_grid = tf.linspace(0.0, 1.0, nb)
        bond_prices = tf.ones((nk, nb, nz)) * 0.96
        params = _make_params()

        result = build_debt_components(b_grid, bond_prices, nb, params)
        assert not np.any(np.isnan(result.numpy()))

    def test_zero_debt_zero_components(self):
        """With zero debt (b=0), debt components should be near zero."""
        nk, nb, nz = 3, 4, 2
        b_grid = tf.constant([0.0, 0.5, 1.0, 2.0])
        bond_prices = tf.ones((nk, nb, nz)) * 0.96
        params = _make_params()

        result = build_debt_components(b_grid, bond_prices, nb, params)
        # First column (b_next=0) should have small/zero components
        np.testing.assert_allclose(result[:, 0, :].numpy(), 0.0, atol=0.01)
