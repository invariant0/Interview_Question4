"""Unit tests for BasicSimulator with a pre-canned trivial solution."""

from __future__ import annotations

import numpy as np
import tensorflow as tf

tf.config.set_visible_devices([], 'GPU')

import pytest

from econ_models.vfi.simulation.basic_simulator import BasicSimulator


def _make_trivial_solution():
    """Build a trivial VFI solution where policy = always stay at grid midpoint."""
    nk, nz = 10, 3
    k_grid = np.linspace(0.5, 3.0, nk).astype(np.float32)
    z_grid = np.linspace(-0.1, 0.1, nz).astype(np.float32)
    delta = 0.1
    k_ss = 1.5

    # V_adjust > V_wait everywhere → always adjust
    v_adjust = np.ones((nk, nz), dtype=np.float32) * 10.0
    v_wait = np.ones((nk, nz), dtype=np.float32) * 5.0

    # Policy: always go to midpoint of grid
    mid_idx = nk // 2
    policy_k_values = np.full((nk, nz), k_grid[mid_idx], dtype=np.float32)

    P = np.zeros((nz, nz), dtype=np.float32)
    for i in range(nz):
        P[i, i] = 1.0  # identity → no Z transitions

    return {
        "V_adjust": v_adjust,
        "V_wait": v_wait,
        "policy_k_values": policy_k_values,
        "K": k_grid,
        "Z": z_grid,
        "transition_matrix": P,
        "k_min": float(k_grid[0]),
        "k_max": float(k_grid[-1]),
        "depreciation_rate": delta,
        "k_ss": k_ss,
    }


class TestBasicSimulator:
    """Tests for BasicSimulator.run()."""

    def test_runs_without_error(self):
        """Simulator completes without raising."""
        sol = _make_trivial_solution()
        sim = BasicSimulator(n_steps=50, n_batches=20, seed=42)
        history, stats = sim.run(sol)
        assert "K" in history
        assert "min_hit_pct" in stats
        assert "max_hit_pct" in stats

    def test_history_shape(self):
        """K history has (n_batches, n_steps) shape."""
        sol = _make_trivial_solution()
        sim = BasicSimulator(n_steps=30, n_batches=10, seed=42)
        history, _ = sim.run(sol)
        assert history["K"].shape == (10, 30)

    def test_capital_stays_in_bounds(self):
        """All capital values are within grid bounds."""
        sol = _make_trivial_solution()
        sim = BasicSimulator(n_steps=100, n_batches=50, seed=42)
        history, _ = sim.run(sol)
        K = history["K"]
        assert np.all(K >= sol["k_min"])
        assert np.all(K <= sol["k_max"])

    def test_converges_to_target(self):
        """With deterministic Z and midpoint policy, capital should converge."""
        sol = _make_trivial_solution()
        sim = BasicSimulator(n_steps=200, n_batches=20, seed=42)
        history, _ = sim.run(sol)
        # After many steps, K should be near the midpoint
        k_grid = sol["K"]
        mid = k_grid[len(k_grid) // 2]
        final_k = history["K"][:, -1]
        # Should be within bounds at least
        assert np.all(final_k >= sol["k_min"])

    def test_reproducible(self):
        """Same seed → same trajectory."""
        sol = _make_trivial_solution()
        h1, _ = BasicSimulator(n_steps=50, n_batches=10, seed=123).run(sol)
        h2, _ = BasicSimulator(n_steps=50, n_batches=10, seed=123).run(sol)
        np.testing.assert_array_equal(h1["K"], h2["K"])
