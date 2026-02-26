"""Unit tests for grid_builder: GridBuilder."""

from __future__ import annotations

import numpy as np
import tensorflow as tf

tf.config.set_visible_devices([], 'GPU')

import pytest

from econ_models.vfi.grids.grid_builder import GridBuilder
from econ_models.config.economic_params import EconomicParams
from econ_models.config.vfi_config import GridConfig


def _make_params(**overrides):
    defaults = dict(
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
    defaults.update(overrides)
    return EconomicParams(**defaults)


class TestGridBuilderProductivity:
    """Tests for productivity grid construction."""

    def test_shape(self):
        """Grid and transition matrix have correct shapes."""
        params = _make_params()
        config = GridConfig(
            n_capital=10, n_productivity=5,
            n_debt=10, tauchen_width=3.0,
        )
        z_grid, P, z_min, z_max = GridBuilder.build_productivity_grid(config, params)
        assert z_grid.shape == (5,)
        assert P.shape == (5, 5)

    def test_transition_rows_sum_to_one(self):
        """Each row of the transition matrix sums to 1."""
        params = _make_params()
        config = GridConfig(
            n_capital=10, n_productivity=7,
            n_debt=10, tauchen_width=3.0,
        )
        _, P, _, _ = GridBuilder.build_productivity_grid(config, params)
        row_sums = tf.reduce_sum(P, axis=1).numpy()
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)

    def test_grid_sorted(self):
        """Productivity grid is sorted ascending."""
        params = _make_params()
        config = GridConfig(
            n_capital=10, n_productivity=11,
            n_debt=10, tauchen_width=3.0,
        )
        z_grid, _, _, _ = GridBuilder.build_productivity_grid(config, params)
        z = z_grid.numpy()
        assert np.all(np.diff(z) > 0)

    def test_min_max_consistent(self):
        """z_min < z_max and match grid endpoints."""
        params = _make_params()
        config = GridConfig(
            n_capital=10, n_productivity=9,
            n_debt=10, tauchen_width=3.0,
        )
        z_grid, _, z_min, z_max = GridBuilder.build_productivity_grid(config, params)
        z = z_grid.numpy()
        assert z_min == pytest.approx(z[0], abs=1e-6)
        assert z_max == pytest.approx(z[-1], abs=1e-6)
        assert z_min < z_max


class TestGridBuilderCapital:
    """Tests for capital grid construction."""

    def test_shape(self):
        """Capital grid has correct number of points."""
        params = _make_params()
        config = GridConfig(
            n_capital=20, n_productivity=5,
            n_debt=10, tauchen_width=3.0,
        )
        k_grid, k_ss = GridBuilder.build_capital_grid(
            config, params, custom_bounds=(0.1, 5.0)
        )
        assert k_grid.shape == (20,)
        assert k_ss > 0

    def test_bounds(self):
        """Grid respects custom bounds."""
        params = _make_params()
        config = GridConfig(
            n_capital=50, n_productivity=5,
            n_debt=10, tauchen_width=3.0,
        )
        k_grid, _ = GridBuilder.build_capital_grid(
            config, params, custom_bounds=(0.5, 3.0)
        )
        k = k_grid.numpy()
        assert k[0] == pytest.approx(0.5, abs=1e-3)
        assert k[-1] == pytest.approx(3.0, abs=1e-3)

    def test_sorted(self):
        """Capital grid is sorted ascending."""
        params = _make_params()
        config = GridConfig(
            n_capital=30, n_productivity=5,
            n_debt=10, tauchen_width=3.0,
        )
        k_grid, _ = GridBuilder.build_capital_grid(
            config, params, custom_bounds=(0.1, 5.0)
        )
        assert np.all(np.diff(k_grid.numpy()) > 0)
