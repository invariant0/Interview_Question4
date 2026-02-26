"""Integration test: basic model VFI solve on a tiny grid.

Verifies end-to-end convergence, value monotonicity, and result dict.
Runs on CPU — no GPU required.
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf

tf.config.set_visible_devices([], 'GPU')

import pytest

from econ_models.config.economic_params import EconomicParams
from econ_models.config.vfi_config import GridConfig
from econ_models.vfi.basic import BasicModelVFI


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


@pytest.fixture(scope="module")
def basic_solution():
    """Solve the basic model once and share across tests."""
    params = _make_params()
    config = GridConfig(
        n_capital=10,
        n_productivity=3,
        tauchen_width=3.0,
    )
    solver = BasicModelVFI(params, config, k_bounds=(0.3, 4.0))
    return solver.solve()


class TestBasicIntegration:
    """End-to-end tests for BasicModelVFI.solve()."""

    def test_result_keys(self, basic_solution):
        """Solution dict contains all expected keys."""
        required = {
            "V", "V_adjust", "V_wait", "K", "Z",
            "policy_adjust_idx", "policy_k_values",
            "transition_matrix", "k_ss", "k_min", "k_max",
            "z_min", "z_max", "depreciation_rate",
        }
        assert required.issubset(set(basic_solution.keys()))

    def test_value_function_shape(self, basic_solution):
        """V has (n_k, n_z) shape."""
        V = basic_solution["V"]
        K = basic_solution["K"]
        Z = basic_solution["Z"]
        assert V.shape == (len(K), len(Z))

    def test_value_nonnegative(self, basic_solution):
        """Value function is non-negative (default floor is 0)."""
        V = basic_solution["V"]
        assert np.all(V >= -1e-6)

    def test_value_monotone_in_capital(self, basic_solution):
        """For each z, V should be non-decreasing in k (at least roughly)."""
        V = basic_solution["V"]
        # Check each z-column: allow small violations due to numeric noise
        for z_idx in range(V.shape[1]):
            diffs = np.diff(V[:, z_idx])
            # Most diffs should be >= 0
            n_decrease = np.sum(diffs < -1e-3)
            assert n_decrease <= 1, (
                f"Too many V decreases in capital at z_idx={z_idx}"
            )

    def test_value_monotone_in_productivity(self, basic_solution):
        """For each k, V should be non-decreasing in z."""
        V = basic_solution["V"]
        for k_idx in range(V.shape[0]):
            diffs = np.diff(V[k_idx, :])
            n_decrease = np.sum(diffs < -1e-3)
            assert n_decrease <= 1, (
                f"Too many V decreases in productivity at k_idx={k_idx}"
            )

    def test_policy_in_bounds(self, basic_solution):
        """Policy K' values are within grid bounds."""
        pk = basic_solution["policy_k_values"]
        k_min = basic_solution["k_min"]
        k_max = basic_solution["k_max"]
        assert np.all(pk >= k_min - 1e-6)
        assert np.all(pk <= k_max + 1e-6)

    def test_transition_matrix_stochastic(self, basic_solution):
        """Transition matrix rows sum to 1."""
        P = basic_solution["transition_matrix"]
        np.testing.assert_allclose(np.sum(P, axis=1), 1.0, atol=1e-6)

    def test_grid_metadata(self, basic_solution):
        """k_ss, k_min, k_max are sensible."""
        assert basic_solution["k_min"] > 0
        assert basic_solution["k_max"] > basic_solution["k_min"]
        assert basic_solution["k_ss"] > 0
