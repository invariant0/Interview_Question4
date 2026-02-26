"""Integration test: solve risky model → simulate → verify no crash.

Round-trip test that exercises the full pipeline on a tiny grid.
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf

tf.config.set_visible_devices([], 'GPU')

import pytest

from econ_models.config.economic_params import EconomicParams
from econ_models.config.vfi_config import GridConfig
from econ_models.vfi.risky import RiskyDebtModelVFI
from econ_models.vfi.simulation.risky_simulator import RiskySimulator


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
def round_trip_data():
    """Solve and simulate once; share across tests."""
    params = _make_params()
    config = GridConfig(
        n_capital=8,
        n_debt=6,
        n_productivity=3,
        tauchen_width=3.0,
        tol_vfi=1e-4,
        max_iter_vfi=200,
        tol_outer=1e-3,
        max_outer=30,
        relax_q=0.5,
    )
    solver = RiskyDebtModelVFI(
        params, config,
        k_bounds=(0.3, 4.0),
        b_bounds=(0.0, 2.0),
        k_chunk_size=8, b_chunk_size=6,
        kp_chunk_size=8, bp_chunk_size=6,
    )
    solution = solver.solve()
    sim = RiskySimulator(params, n_steps=50, n_batches=20, seed=42)
    history, stats = sim.run(solution)
    return solution, history, stats


class TestRoundTrip:
    """Solve → Simulate round-trip tests."""

    def test_simulation_completes(self, round_trip_data):
        """Simulation runs without error."""
        _, history, stats = round_trip_data
        assert "K" in history
        assert "B" in history

    def test_history_shape(self, round_trip_data):
        """History arrays have expected shapes."""
        _, history, _ = round_trip_data
        assert history["K"].shape == (20, 50)
        assert history["B"].shape == (20, 50)

    def test_capital_in_grid_range(self, round_trip_data):
        """Simulated capital is within original grid bounds."""
        solution, history, _ = round_trip_data
        K = history["K"]
        k_min = float(solution["K"][0])
        k_max = float(solution["K"][-1])
        assert np.all(K >= k_min - 1e-6)
        assert np.all(K <= k_max + 1e-6)

    def test_debt_in_grid_range(self, round_trip_data):
        """Simulated debt is within original grid bounds."""
        solution, history, _ = round_trip_data
        B = history["B"]
        b_min = float(solution["B"][0])
        b_max = float(solution["B"][-1])
        assert np.all(B >= b_min - 1e-6)
        assert np.all(B <= b_max + 1e-6)

    def test_stats_sensible(self, round_trip_data):
        """Stats contain expected keys and are finite."""
        _, _, stats = round_trip_data
        assert "k_min_hit_pct" in stats
        assert "k_max_hit_pct" in stats
        assert "default_rate" in stats
        for k, v in stats.items():
            assert np.isfinite(v), f"stat '{k}' is not finite: {v}"

    def test_no_nan_in_history(self, round_trip_data):
        """No NaN in simulated histories."""
        _, history, _ = round_trip_data
        assert not np.any(np.isnan(history["K"]))
        assert not np.any(np.isnan(history["B"]))
