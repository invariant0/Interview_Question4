"""Unit tests for RiskySimulator with pre-canned scenarios."""

from __future__ import annotations

import numpy as np
import tensorflow as tf

tf.config.set_visible_devices([], 'GPU')

import pytest

from econ_models.vfi.simulation.risky_simulator import RiskySimulator
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


def _make_never_default_solution():
    """Build a solution where no firm ever defaults."""
    nk, nb, nz = 8, 6, 3
    k_grid = np.linspace(0.5, 3.0, nk).astype(np.float32)
    b_grid = np.linspace(0.0, 1.0, nb).astype(np.float32)
    z_grid = np.linspace(-0.1, 0.1, nz).astype(np.float32)

    P = np.eye(nz, dtype=np.float32)  # No transitions

    # Never default
    pol_default = np.zeros((nk, nb, nz), dtype=bool)
    # Always adjust
    pol_adjust = np.ones((nk, nb, nz), dtype=bool)
    # Stay at midpoint
    mid_k, mid_b = nk // 2, nb // 2
    pol_k_idx = np.full((nk, nb, nz), mid_k, dtype=np.int32)
    pol_b_idx = np.full((nk, nb, nz), mid_b, dtype=np.int32)
    pol_bw_idx = np.full((nk, nb, nz), mid_b, dtype=np.int32)

    return {
        "K": k_grid, "B": b_grid, "Z": z_grid,
        "transition_matrix": P,
        "policy_default": pol_default,
        "policy_adjust": pol_adjust,
        "policy_k_idx": pol_k_idx,
        "policy_b_idx": pol_b_idx,
        "policy_b_wait_idx": pol_bw_idx,
    }


def _make_always_default_solution():
    """Build a solution where every firm defaults every period."""
    nk, nb, nz = 6, 4, 2
    k_grid = np.linspace(0.5, 3.0, nk).astype(np.float32)
    b_grid = np.linspace(0.0, 1.0, nb).astype(np.float32)
    z_grid = np.linspace(-0.1, 0.1, nz).astype(np.float32)

    P = np.eye(nz, dtype=np.float32)

    pol_default = np.ones((nk, nb, nz), dtype=bool)
    pol_adjust = np.zeros((nk, nb, nz), dtype=bool)
    pol_k_idx = np.zeros((nk, nb, nz), dtype=np.int32)
    pol_b_idx = np.zeros((nk, nb, nz), dtype=np.int32)
    pol_bw_idx = np.zeros((nk, nb, nz), dtype=np.int32)

    return {
        "K": k_grid, "B": b_grid, "Z": z_grid,
        "transition_matrix": P,
        "policy_default": pol_default,
        "policy_adjust": pol_adjust,
        "policy_k_idx": pol_k_idx,
        "policy_b_idx": pol_b_idx,
        "policy_b_wait_idx": pol_bw_idx,
    }


class TestDepreciatedKIndexMap:
    """Tests for _build_depreciated_k_index_map."""

    def test_monotonic(self):
        """Depreciated indices are non-decreasing."""
        k_grid = np.linspace(0.5, 3.0, 20)
        idx = RiskySimulator._build_depreciated_k_index_map(k_grid, delta=0.1)
        assert np.all(np.diff(idx) >= 0)

    def test_within_bounds(self):
        """All indices are valid grid indices."""
        k_grid = np.linspace(0.5, 3.0, 15)
        idx = RiskySimulator._build_depreciated_k_index_map(k_grid, delta=0.1)
        assert np.all(idx >= 0)
        assert np.all(idx < len(k_grid))

    def test_zero_depreciation(self):
        """With δ=0, indices should be identity."""
        k_grid = np.linspace(0.5, 3.0, 10)
        idx = RiskySimulator._build_depreciated_k_index_map(k_grid, delta=0.0)
        np.testing.assert_array_equal(idx, np.arange(len(k_grid)))


class TestRiskySimulatorNeverDefault:
    """Tests with never-default scenario."""

    def test_runs_without_error(self):
        """Simulation completes."""
        params = _make_params()
        sol = _make_never_default_solution()
        sim = RiskySimulator(params, n_steps=30, n_batches=10, seed=42)
        history, stats = sim.run(sol)
        assert "K" in history
        assert "B" in history

    def test_zero_default_rate(self):
        """No defaults should occur."""
        params = _make_params()
        sol = _make_never_default_solution()
        sim = RiskySimulator(params, n_steps=50, n_batches=20, seed=42)
        _, stats = sim.run(sol)
        assert stats["default_rate"] == 0.0

    def test_history_shape(self):
        """History shape is (n_batches, n_steps)."""
        params = _make_params()
        sol = _make_never_default_solution()
        sim = RiskySimulator(params, n_steps=40, n_batches=15, seed=42)
        history, _ = sim.run(sol)
        assert history["K"].shape == (15, 40)
        assert history["B"].shape == (15, 40)


class TestRiskySimulatorAlwaysDefault:
    """Tests with always-default scenario."""

    def test_high_default_rate(self):
        """Default rate should be very high (every period)."""
        params = _make_params()
        sol = _make_always_default_solution()
        sim = RiskySimulator(params, n_steps=50, n_batches=20, seed=42)
        _, stats = sim.run(sol)
        # Every observation is a default
        assert stats["default_rate"] > 90.0

    def test_reproducible(self):
        """Same seed → same results."""
        params = _make_params()
        sol = _make_always_default_solution()
        h1, s1 = RiskySimulator(params, n_steps=30, n_batches=10, seed=99).run(sol)
        h2, s2 = RiskySimulator(params, n_steps=30, n_batches=10, seed=99).run(sol)
        np.testing.assert_array_equal(h1["K"], h2["K"])
        assert s1["default_rate"] == s2["default_rate"]
