"""Integration test: risky-debt model VFI solve on a tiny grid.

Verifies end-to-end convergence, value non-negativity, and bond-price bounds.
Runs on CPU — no GPU required.
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf

tf.config.set_visible_devices([], 'GPU')

import pytest

from econ_models.config.economic_params import EconomicParams
from econ_models.config.vfi_config import GridConfig
from econ_models.vfi.risky import RiskyDebtModelVFI


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
def risky_solution():
    """Solve the risky model once on a tiny grid; share across tests."""
    params = _make_params()
    config = GridConfig(
        n_capital=8,
        n_debt=6,
        n_productivity=3,
        tauchen_width=3.0,
        tol_vfi=1e-4,      # looser for speed
        max_iter_vfi=200,
        tol_outer=1e-3,
        max_outer=30,
        relax_q=0.5,
    )
    solver = RiskyDebtModelVFI(
        params, config,
        k_bounds=(0.3, 4.0),
        b_bounds=(0.0, 2.0),
        k_chunk_size=8,
        b_chunk_size=6,
        kp_chunk_size=8,
        bp_chunk_size=6,
    )
    return solver.solve()


class TestRiskyIntegration:
    """End-to-end tests for RiskyDebtModelVFI.solve()."""

    def test_result_keys(self, risky_solution):
        """Solution dict contains expected keys."""
        required = {
            "V", "K", "B", "Z", "Q", "transition_matrix",
            "policy_default", "policy_adjust",
            "policy_k_idx", "policy_b_idx", "policy_b_wait_idx",
            "policy_k_values", "policy_b_values", "policy_b_wait_values",
        }
        assert required.issubset(set(risky_solution.keys()))

    def test_value_function_shape(self, risky_solution):
        """V has (nk, nb, nz) shape."""
        V = risky_solution["V"]
        nk = len(risky_solution["K"])
        nb = len(risky_solution["B"])
        nz = len(risky_solution["Z"])
        assert V.shape == (nk, nb, nz)

    def test_value_nonnegative(self, risky_solution):
        """Value function is non-negative (default floor is 0)."""
        V = risky_solution["V"]
        assert np.all(V >= -1e-6)

    def test_bond_prices_bounded(self, risky_solution):
        """Bond prices are in [0, q_rf]."""
        q = risky_solution["Q"]
        q_rf = 1.0 / (1.0 + 0.04)
        assert np.all(q >= -1e-6), f"Min bond price: {np.min(q)}"
        assert np.all(q <= q_rf + 0.05), f"Max bond price: {np.max(q)}"

    def test_default_region_boolean(self, risky_solution):
        """policy_default is boolean."""
        pd = risky_solution["policy_default"]
        assert pd.dtype == bool or pd.dtype == np.bool_

    def test_policy_indices_valid(self, risky_solution):
        """Non-default policy indices are within grid bounds."""
        pd = risky_solution["policy_default"]
        pk = risky_solution["policy_k_idx"]
        pb = risky_solution["policy_b_idx"]
        nk = len(risky_solution["K"])
        nb = len(risky_solution["B"])

        valid = ~pd
        if np.any(valid):
            assert np.all(pk[valid] >= 0)
            assert np.all(pk[valid] < nk)
            assert np.all(pb[valid] >= 0)
            assert np.all(pb[valid] < nb)

    def test_transition_matrix_stochastic(self, risky_solution):
        """Each row of P sums to 1."""
        P = risky_solution["transition_matrix"]
        np.testing.assert_allclose(np.sum(P, axis=1), 1.0, atol=1e-6)
