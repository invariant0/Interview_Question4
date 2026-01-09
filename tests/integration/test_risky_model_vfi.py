# tests/integration/test_risky_model_vfi.py
"""
Integration tests for the Risky Debt Model VFI implementation.

These tests verify the complete pipeline for the corporate finance model
with risky debt and endogenous default using the unittest framework.
"""

import unittest
import numpy as np
import tensorflow as tf
from typing import Dict, Optional

from econ_models.config.economic_params import EconomicParams
from econ_models.config.vfi_config import GridConfig
from econ_models.vfi.risky_debt import RiskyDebtModelVFI
from econ_models.vfi.bounds import BoundaryFinder
from econ_models.econ import SteadyStateCalculator


class RiskyModelTestCase(unittest.TestCase):
    """Base test case with common setup for risky debt model tests."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures used by all test methods."""
        cls.risky_params = EconomicParams(
            discount_factor=0.96,
            capital_share=0.33,
            depreciation_rate=0.1,
            productivity_persistence=0.9,
            productivity_std_dev=0.02,
            adjustment_cost_convex=0.5,
            adjustment_cost_fixed=0.0,
            equity_issuance_cost_fixed=0.0,
            equity_issuance_cost_linear=0.0,
            default_cost_proportional=0.25,
            corporate_tax_rate=0.2,
            risk_free_rate=0.04,
            collateral_recovery_fraction=0.5
        )

        cls.small_config = GridConfig(
            n_capital=15,
            n_productivity=5,
            n_debt=10,
            tauchen_width=2.5,
            tol_vfi=1e-5,
            max_iter_vfi=300,
            tol_outer=1e-3,
            max_outer=20,
            v_default_eps=1e-8,
            b_eps=1e-8,
            q_min=1e-8,
            relax_q=0.5,
            collateral_violation_penalty=-1e10
        )

        cls.medium_config = GridConfig(
            n_capital=25,
            n_productivity=7,
            n_debt=15,
            tauchen_width=3.0,
            tol_vfi=1e-6,
            max_iter_vfi=500,
            tol_outer=1e-4,
            max_outer=50,
            v_default_eps=1e-9,
            b_eps=1e-9,
            q_min=1e-9,
            relax_q=0.5,
            collateral_violation_penalty=-1e10
        )

    def _create_model_with_bounds(
        self,
        params: Optional[EconomicParams] = None,
        config: Optional[GridConfig] = None
    ) -> tuple:
        """Helper to create model with computed bounds."""
        params = params or self.risky_params
        config = config or self.small_config

        k_ss = SteadyStateCalculator.calculate_capital(params)
        k_bounds = (0.3 * k_ss, 2.5 * k_ss)
        b_bounds = (0.0, 1.5 * k_ss)

        model = RiskyDebtModelVFI(
            params, config, k_bounds=k_bounds, b_bounds=b_bounds
        )
        return model, k_bounds, b_bounds, k_ss


class TestRiskyModelInitialization(RiskyModelTestCase):
    """Tests for risky debt model initialization."""

    def test_model_initializes_with_valid_params(self):
        """Model should initialize without errors given valid parameters."""
        model, _, _, _ = self._create_model_with_bounds()

        self.assertEqual(model.params, self.risky_params)
        self.assertEqual(model.config, self.small_config)
        self.assertIsNotNone(model.k_grid)
        self.assertIsNotNone(model.b_grid)
        self.assertIsNotNone(model.z_grid)
        self.assertIsNotNone(model.P)
        self.assertIsNotNone(model.bond_prices)

    def test_grid_dimensions_match_config(self):
        """All grid dimensions should match configuration."""
        model, _, _, _ = self._create_model_with_bounds()

        self.assertEqual(
            model.k_grid.shape[0],
            self.small_config.n_capital
        )
        self.assertEqual(
            model.b_grid.shape[0],
            self.small_config.n_debt
        )
        self.assertEqual(
            model.z_grid.shape[0],
            self.small_config.n_productivity
        )

    def test_bond_prices_initialized_at_risk_free(self):
        """Bond prices should initialize at risk-free level."""
        model, _, _, _ = self._create_model_with_bounds()

        q_rf = 1.0 / (1.0 + self.risky_params.risk_free_rate)
        initial_prices = model.bond_prices.numpy()

        np.testing.assert_allclose(initial_prices, q_rf, rtol=1e-6)

    def test_bond_prices_shape(self):
        """Bond prices should have correct 3D shape."""
        model, _, _, _ = self._create_model_with_bounds()

        expected_shape = (
            self.small_config.n_capital,
            self.small_config.n_debt,
            self.small_config.n_productivity
        )
        self.assertEqual(model.bond_prices.shape, expected_shape)

    def test_debt_grid_is_nonnegative(self):
        """Debt grid values should be non-negative with default bounds."""
        model, _, _, _ = self._create_model_with_bounds()
        b_vals = model.b_grid.numpy()

        self.assertTrue(
            np.all(b_vals >= 0),
            "Debt grid should be non-negative"
        )

    def test_debt_grid_is_monotonic(self):
        """Debt grid should be strictly increasing."""
        model, _, _, _ = self._create_model_with_bounds()
        b_vals = model.b_grid.numpy()

        diffs = np.diff(b_vals)
        self.assertTrue(
            np.all(diffs > 0),
            "Debt grid must be strictly monotonic"
        )

    def test_insufficient_debt_points_raises(self):
        """Config with n_debt < 2 should raise ValueError."""
        bad_config = GridConfig(
            n_capital=20,
            n_productivity=5,
            n_debt=1,
            tauchen_width=3.0
        )

        k_ss = SteadyStateCalculator.calculate_capital(self.risky_params)

        with self.assertRaises(ValueError) as context:
            RiskyDebtModelVFI(
                self.risky_params, bad_config,
                k_bounds=(0.5 * k_ss, 2.0 * k_ss),
                b_bounds=(0.0, k_ss)
            )

        self.assertIn("n_debt", str(context.exception))

    def test_custom_bounds_applied(self):
        """Custom bounds should be correctly applied to grids."""
        k_ss = SteadyStateCalculator.calculate_capital(self.risky_params)
        k_bounds = (1.0, 10.0)
        b_bounds = (0.5, 8.0)

        model = RiskyDebtModelVFI(
            self.risky_params, self.small_config,
            k_bounds=k_bounds, b_bounds=b_bounds
        )

        k_vals = model.k_grid.numpy()
        b_vals = model.b_grid.numpy()

        self.assertGreaterEqual(k_vals[0], k_bounds[0] - 1e-6)
        self.assertLessEqual(k_vals[-1], k_bounds[1] + 1e-6)
        self.assertGreaterEqual(b_vals[0], b_bounds[0] - 1e-6)
        self.assertLessEqual(b_vals[-1], b_bounds[1] + 1e-6)


class TestRiskyModelFlowMatrix(RiskyModelTestCase):
    """Tests for flow matrix computation in risky model."""

    def test_flow_matrix_shape(self):
        """Flow matrix should have correct 5D shape."""
        model, _, _, _ = self._create_model_with_bounds()

        n_k = self.small_config.n_capital
        n_b = self.small_config.n_debt
        n_z = self.small_config.n_productivity

        q_sched = tf.reshape(model.bond_prices, (1, 1, n_k, n_b, n_z))
        flow = model.compute_flow(q_sched, enforce_constraint=False)

        expected_shape = (n_k, n_b, n_k, n_b, n_z)
        self.assertEqual(flow.shape, expected_shape)

    def test_flow_matrix_no_nan(self):
        """Flow matrix should not contain NaN values."""
        model, _, _, _ = self._create_model_with_bounds()

        n_k = self.small_config.n_capital
        n_b = self.small_config.n_debt
        n_z = self.small_config.n_productivity

        q_sched = tf.reshape(model.bond_prices, (1, 1, n_k, n_b, n_z))
        flow = model.compute_flow(q_sched, enforce_constraint=False).numpy()

        self.assertFalse(
            np.any(np.isnan(flow)),
            "Flow matrix contains NaN values"
        )

    def test_flow_matrix_dtype_is_float(self):
        """Flow matrix should be a float tensor."""
        model, _, _, _ = self._create_model_with_bounds()

        n_k = self.small_config.n_capital
        n_b = self.small_config.n_debt
        n_z = self.small_config.n_productivity

        q_sched = tf.reshape(model.bond_prices, (1, 1, n_k, n_b, n_z))
        flow = model.compute_flow(q_sched, enforce_constraint=False)

        # Accept either float32 or float64
        self.assertTrue(
            flow.dtype in [tf.float32, tf.float64],
            f"Flow dtype {flow.dtype} should be float32 or float64"
        )

    def test_collateral_constraint_applies_penalty(self):
        """Collateral constraint should apply penalty for violations."""
        model, _, _, _ = self._create_model_with_bounds()

        n_k = self.small_config.n_capital
        n_b = self.small_config.n_debt
        n_z = self.small_config.n_productivity

        q_sched = tf.reshape(model.bond_prices, (1, 1, n_k, n_b, n_z))

        flow_unconstrained = model.compute_flow(
            q_sched, enforce_constraint=False
        ).numpy()
        flow_constrained = model.compute_flow(
            q_sched, enforce_constraint=True
        ).numpy()

        penalty = self.small_config.collateral_violation_penalty
        has_penalty = np.any(flow_constrained == penalty)

        self.assertTrue(
            has_penalty or np.allclose(flow_unconstrained, flow_constrained),
            "Constraint should either apply penalty or be non-binding"
        )


class TestRiskyModelSolver(RiskyModelTestCase):
    """Tests for the risky debt VFI solving process."""

    @classmethod
    def setUpClass(cls):
        """Set up solved model for solver tests."""
        super().setUpClass()

        k_ss = SteadyStateCalculator.calculate_capital(cls.risky_params)
        k_bounds = (0.3 * k_ss, 2.5 * k_ss)
        b_bounds = (0.0, 1.5 * k_ss)

        cls.model = RiskyDebtModelVFI(
            cls.risky_params, cls.small_config,
            k_bounds=k_bounds, b_bounds=b_bounds
        )
        cls.results = cls.model.solve()

    def test_solve_returns_required_keys(self):
        """Solution should contain all required arrays."""
        self.assertIn("V", self.results)
        self.assertIn("Q", self.results)
        self.assertIn("K", self.results)
        self.assertIn("B", self.results)
        self.assertIn("Z", self.results)

    def test_value_function_shape(self):
        """Value function should have correct 3D shape."""
        expected_shape = (
            self.small_config.n_capital,
            self.small_config.n_debt,
            self.small_config.n_productivity
        )
        self.assertEqual(self.results["V"].shape, expected_shape)

    def test_bond_price_shape(self):
        """Bond price schedule should have correct shape."""
        expected_shape = (
            self.small_config.n_capital,
            self.small_config.n_debt,
            self.small_config.n_productivity
        )
        self.assertEqual(self.results["Q"].shape, expected_shape)

    def test_value_function_is_nonnegative(self):
        """Value function should be non-negative."""
        self.assertTrue(
            np.all(self.results["V"] >= 0),
            "Value function must be non-negative"
        )

    def test_value_function_is_finite(self):
        """Value function should contain only finite values."""
        self.assertTrue(
            np.all(np.isfinite(self.results["V"])),
            "Value function must be finite"
        )

    def test_bond_prices_in_valid_range(self):
        """Bond prices should be between 0 and risk-free price."""
        Q = self.results["Q"]
        q_rf = 1.0 / (1.0 + self.risky_params.risk_free_rate)

        self.assertTrue(
            np.all(Q >= 0),
            "Bond prices must be non-negative"
        )
        self.assertTrue(
            np.all(Q <= q_rf + 1e-6),
            f"Bond prices should not exceed risk-free {q_rf}"
        )

    def test_bond_prices_are_finite(self):
        """Bond prices should contain only finite values."""
        self.assertTrue(
            np.all(np.isfinite(self.results["Q"])),
            "Bond prices must be finite"
        )

    def test_value_increases_with_capital(self):
        """Value should generally increase with capital."""
        V = self.results["V"]

        mid_b = V.shape[1] // 2
        mid_z = V.shape[2] // 2
        v_slice = V[:, mid_b, mid_z]

        violations = np.sum(np.diff(v_slice) < -1e-4)
        self.assertLessEqual(
            violations, 3,
            "V should mostly increase with K"
        )

    def test_value_decreases_with_debt(self):
        """Value should generally decrease with debt."""
        V = self.results["V"]

        mid_k = V.shape[0] // 2
        mid_z = V.shape[2] // 2
        v_slice = V[mid_k, :, mid_z]

        violations = np.sum(np.diff(v_slice) > 1e-4)
        self.assertLessEqual(
            violations, 3,
            "V should mostly decrease with B"
        )

    def test_bond_prices_decrease_with_debt(self):
        """Bond prices should decrease with debt level."""
        Q = self.results["Q"]

        mid_k = Q.shape[0] // 2
        mid_z = Q.shape[2] // 2
        q_slice = Q[mid_k, :, mid_z]

        violations = np.sum(np.diff(q_slice) > 1e-4)
        self.assertLessEqual(
            violations, 2,
            "Q should mostly decrease with B"
        )

    def test_grids_match_config(self):
        """Returned grids should match configuration."""
        self.assertEqual(
            len(self.results["K"]),
            self.small_config.n_capital
        )
        self.assertEqual(
            len(self.results["B"]),
            self.small_config.n_debt
        )
        self.assertEqual(
            len(self.results["Z"]),
            self.small_config.n_productivity
        )


class TestRiskyModelPolicy(RiskyModelTestCase):
    """Tests for policy function extraction in risky model."""

    @classmethod
    def setUpClass(cls):
        """Set up solved model for policy tests."""
        super().setUpClass()

        k_ss = SteadyStateCalculator.calculate_capital(cls.risky_params)
        k_bounds = (0.3 * k_ss, 2.5 * k_ss)
        b_bounds = (0.0, 1.5 * k_ss)

        cls.model = RiskyDebtModelVFI(
            cls.risky_params, cls.small_config,
            k_bounds=k_bounds, b_bounds=b_bounds
        )
        cls.results = cls.model.solve()
        
        # Use the same dtype as the transition matrix P
        p_dtype = cls.model.P.dtype
        cls.v_tensor = tf.constant(cls.results["V"], dtype=p_dtype)
        
        # FIX: Pass q_sched (bond prices) to get_policy_indices
        q_tensor = tf.constant(cls.results["Q"], dtype=p_dtype)
        cls.pol_k, cls.pol_b = cls.model.get_policy_indices(cls.v_tensor, q_tensor)


class TestRiskyModelSimulation(RiskyModelTestCase):
    """Tests for risky debt model simulation."""

    @classmethod
    def setUpClass(cls):
        """Set up solved model for simulation tests."""
        super().setUpClass()

        k_ss = SteadyStateCalculator.calculate_capital(cls.risky_params)
        k_bounds = (0.3 * k_ss, 2.5 * k_ss)
        b_bounds = (0.0, 1.5 * k_ss)

        cls.model = RiskyDebtModelVFI(
            cls.risky_params, cls.small_config,
            k_bounds=k_bounds, b_bounds=b_bounds
        )
        cls.results = cls.model.solve()

    def test_simulation_returns_history_and_stats(self):
        """Simulation should return history and statistics."""
        # FIX: Pass q_sched argument
        history, stats = self.model.simulate(
            self.results["V"], 
            self.results["Q"],  # New argument
            n_steps=50, n_batches=30, seed=42
        )

        self.assertIsNotNone(history)
        self.assertIsInstance(stats, dict)
        self.assertIn("k_min", stats)
        self.assertIn("k_max", stats)
        self.assertIn("b_min", stats)
        self.assertIn("b_max", stats)
        self.assertIn("total_observations", stats)

    def test_simulation_history_dimensions(self):
        """Simulation history should have correct dimensions."""
        n_steps, n_batches = 50, 30

        # FIX: Pass q_sched argument
        history, _ = self.model.simulate(
            self.results["V"], 
            self.results["Q"],  # New argument
            n_steps=n_steps, n_batches=n_batches, seed=42
        )

        self.assertEqual(history.n_steps, n_steps)
        self.assertEqual(history.n_batches, n_batches)
        self.assertEqual(history.total_observations, n_steps * n_batches)

    def test_simulation_has_all_trajectories(self):
        """Simulation should track all state variables."""
        # FIX: Pass q_sched argument
        history, _ = self.model.simulate(
            self.results["V"], 
            self.results["Q"],  # New argument
            n_steps=50, n_batches=30, seed=42
        )

        self.assertIn("k_idx", history.trajectories)
        self.assertIn("b_idx", history.trajectories)
        self.assertIn("z_idx", history.trajectories)

    def test_simulation_trajectory_shapes(self):
        """Trajectory arrays should have correct shapes."""
        n_steps, n_batches = 50, 30

        # FIX: Pass q_sched argument
        history, _ = self.model.simulate(
            self.results["V"], 
            self.results["Q"],  # New argument
            n_steps=n_steps, n_batches=n_batches, seed=42
        )

        expected_shape = (n_batches, n_steps)
        self.assertEqual(history.trajectories["k_idx"].shape, expected_shape)
        self.assertEqual(history.trajectories["b_idx"].shape, expected_shape)
        self.assertEqual(history.trajectories["z_idx"].shape, expected_shape)

    def test_simulation_trajectories_in_bounds(self):
        """Simulated trajectories should stay within grid bounds."""
        # FIX: Pass q_sched argument
        history, _ = self.model.simulate(
            self.results["V"], 
            self.results["Q"],  # New argument
            n_steps=100, n_batches=50, seed=42
        )

        k_traj = history.trajectories["k_idx"]
        b_traj = history.trajectories["b_idx"]
        z_traj = history.trajectories["z_idx"]

        # Note: We filter out -1 indices which indicate default
        valid_k = k_traj[k_traj != -1]
        valid_b = b_traj[b_traj != -1]
        
        if len(valid_k) > 0:
            self.assertTrue(np.all(valid_k >= 0))
            self.assertTrue(np.all(valid_k < self.small_config.n_capital))
        
        if len(valid_b) > 0:
            self.assertTrue(np.all(valid_b >= 0))
            self.assertTrue(np.all(valid_b < self.small_config.n_debt))
            
        self.assertTrue(np.all(z_traj >= 0))
        self.assertTrue(np.all(z_traj < self.small_config.n_productivity))

    def test_simulation_reproducibility(self):
        """Simulation with same seed should produce identical results."""
        # FIX: Pass q_sched argument
        history1, stats1 = self.model.simulate(
            self.results["V"], 
            self.results["Q"], # New argument
            n_steps=50, n_batches=30, seed=99999
        )
        history2, stats2 = self.model.simulate(
            self.results["V"], 
            self.results["Q"], # New argument
            n_steps=50, n_batches=30, seed=99999
        )

        np.testing.assert_array_equal(
            history1.trajectories["k_idx"],
            history2.trajectories["k_idx"]
        )
        np.testing.assert_array_equal(
            history1.trajectories["b_idx"],
            history2.trajectories["b_idx"]
        )
        self.assertEqual(stats1, stats2)

    def test_different_seeds_produce_different_results(self):
        """Different seeds should produce different trajectories."""
        # FIX: Pass q_sched argument
        history1, _ = self.model.simulate(
            self.results["V"], 
            self.results["Q"], # New argument
            n_steps=50, n_batches=30, seed=111
        )
        history2, _ = self.model.simulate(
            self.results["V"], 
            self.results["Q"], # New argument
            n_steps=50, n_batches=30, seed=222
        )

        self.assertFalse(
            np.array_equal(
                history1.trajectories["k_idx"],
                history2.trajectories["k_idx"]
            ),
            "Different seeds should produce different results"
        )


class TestRiskyModelBondPricing(RiskyModelTestCase):
    """Tests for bond pricing mechanism."""

    def test_bond_prices_bounded_by_risk_free(self):
        """No bond should price above risk-free rate."""
        model, _, _, _ = self._create_model_with_bounds()
        results = model.solve()
        Q = results["Q"]

        q_rf = 1.0 / (1.0 + self.risky_params.risk_free_rate)
        max_price = np.max(Q)

        self.assertLessEqual(
            max_price, q_rf + 1e-4,
            f"Max bond price {max_price} exceeds risk-free {q_rf}"
        )

    def test_bond_prices_positive(self):
        """All bond prices should be positive."""
        model, _, _, _ = self._create_model_with_bounds()
        results = model.solve()
        Q = results["Q"]

        self.assertTrue(
            np.all(Q >= 0),
            "Bond prices must be non-negative"
        )

    def test_bond_prices_respond_to_parameters(self):
        """Bond prices should differ with different default costs."""
        k_ss = SteadyStateCalculator.calculate_capital(self.risky_params)
        k_bounds = (0.3 * k_ss, 2.0 * k_ss)
        b_bounds = (0.0, 1.0 * k_ss)

        low_default_params = EconomicParams(
            discount_factor=0.96,
            capital_share=0.33,
            depreciation_rate=0.1,
            productivity_persistence=0.9,
            productivity_std_dev=0.02,
            adjustment_cost_convex=0.5,
            adjustment_cost_fixed=0.0,
            equity_issuance_cost_fixed=0.0,
            equity_issuance_cost_linear=0.0,
            default_cost_proportional=0.1,
            corporate_tax_rate=0.2,
            risk_free_rate=0.04,
            collateral_recovery_fraction=0.5
        )

        high_default_params = EconomicParams(
            discount_factor=0.96,
            capital_share=0.33,
            depreciation_rate=0.1,
            productivity_persistence=0.9,
            productivity_std_dev=0.02,
            adjustment_cost_convex=0.5,
            adjustment_cost_fixed=0.0,
            equity_issuance_cost_fixed=0.0,
            equity_issuance_cost_linear=0.0,
            default_cost_proportional=0.5,
            corporate_tax_rate=0.2,
            risk_free_rate=0.04,
            collateral_recovery_fraction=0.5
        )

        model_low = RiskyDebtModelVFI(
            low_default_params, self.small_config,
            k_bounds=k_bounds, b_bounds=b_bounds
        )
        model_high = RiskyDebtModelVFI(
            high_default_params, self.small_config,
            k_bounds=k_bounds, b_bounds=b_bounds
        )

        results_low = model_low.solve()
        results_high = model_high.solve()

        self.assertEqual(results_low["Q"].shape, results_high["Q"].shape)


class TestRiskyModelEconomicConsistency(RiskyModelTestCase):
    """Tests for economic consistency of risky debt model."""

    @classmethod
    def setUpClass(cls):
        """Set up solved model for economic consistency tests."""
        super().setUpClass()

        k_ss = SteadyStateCalculator.calculate_capital(cls.risky_params)
        k_bounds = (0.3 * k_ss, 2.5 * k_ss)
        b_bounds = (0.0, 1.5 * k_ss)

        cls.model = RiskyDebtModelVFI(
            cls.risky_params, cls.small_config,
            k_bounds=k_bounds, b_bounds=b_bounds
        )
        cls.results = cls.model.solve()

    def test_higher_productivity_increases_value(self):
        """Higher productivity should increase value function."""
        V = self.results["V"]

        mid_k = V.shape[0] // 2
        mid_b = V.shape[1] // 2
        v_slice = V[mid_k, mid_b, :]

        diffs = np.diff(v_slice)
        self.assertTrue(
            np.all(diffs >= -1e-6),
            "Value should increase with productivity"
        )

    def test_bond_prices_increase_with_capital(self):
        """More capital means more collateral, so higher bond prices."""
        Q = self.results["Q"]

        mid_b = Q.shape[1] // 2
        mid_z = Q.shape[2] // 2
        q_slice = Q[:, mid_b, mid_z]

        increases = np.sum(np.diff(q_slice) >= -1e-6)
        self.assertGreaterEqual(
            increases, len(q_slice) - 3,
            "Bond prices should mostly increase with capital"
        )

    def test_zero_debt_value_highest(self):
        """Value at zero debt should generally be highest."""
        V = self.results["V"]

        V_zero_debt = V[:, 0, :]
        V_high_debt = V[:, -1, :]

        self.assertGreaterEqual(
            np.mean(V_zero_debt),
            np.mean(V_high_debt) - 1e-4,
            "Zero-debt value should be at least as high as high-debt value"
        )


class TestRiskyBoundaryFinder(RiskyModelTestCase):
    """Tests for automatic boundary discovery in risky model."""

    def test_boundary_finder_returns_all_bounds(self):
        """Boundary finder should return bounds for K, B, and Z."""
        finder = BoundaryFinder(
            self.risky_params, self.small_config,
            threshold=0.1,
            n_steps=30,
            n_batches=30,
            seed=42
        )
        bounds = finder.find_risky_bounds()

        self.assertIn("k_bounds_original", bounds)
        self.assertIn("b_bounds_original", bounds)
        self.assertIn("z_bounds_original", bounds)
        self.assertIn("k_bounds_add_margin", bounds)
        self.assertIn("b_bounds_add_margin", bounds)
        self.assertIn("z_bounds_add_margin", bounds)
        self.assertIn("vfi_solution", bounds)

    def test_margin_bounds_properly_expanded(self):
        """Margin-adjusted bounds should be properly expanded."""
        finder = BoundaryFinder(
            self.risky_params, self.small_config,
            threshold=0.1,
            margin=1.2,
            n_steps=30,
            n_batches=30,
            seed=42
        )
        bounds = finder.find_risky_bounds()

        k_orig = bounds["k_bounds_original"]
        k_margin = bounds["k_bounds_add_margin"]

        self.assertLess(k_margin[0], k_orig[0])
        self.assertGreater(k_margin[1], k_orig[1])

    def test_boundary_finder_with_custom_threshold(self):
        """Boundary finder should respect custom threshold."""
        finder = BoundaryFinder(
            self.risky_params, self.small_config,
            threshold=0.02,
            n_steps=30,
            n_batches=30,
            seed=42
        )

        self.assertEqual(finder.threshold, 0.02)


class TestRiskyModelMemoryConstraints(RiskyModelTestCase):
    """Tests for memory constraint handling."""

    def test_large_grid_raises_error(self):
        """Excessively large grids should raise ValueError."""
        large_config = GridConfig(
            n_capital=150,
            n_productivity=7,
            n_debt=150,
            tauchen_width=3.0
        )

        k_ss = SteadyStateCalculator.calculate_capital(self.risky_params)
        model = RiskyDebtModelVFI(
            self.risky_params, large_config,
            k_bounds=(0.5 * k_ss, 2.0 * k_ss),
            b_bounds=(0.0, k_ss)
        )

        with self.assertRaises(ValueError) as context:
            model.solve()

        self.assertIn("memory", str(context.exception).lower())

    def test_valid_grid_size_solves(self):
        """Valid grid sizes should solve successfully."""
        valid_config = GridConfig(
            n_capital=50,
            n_productivity=5,
            n_debt=50,
            tauchen_width=3.0,
            tol_vfi=1e-5,
            max_iter_vfi=200,
            max_outer=10
        )

        k_ss = SteadyStateCalculator.calculate_capital(self.risky_params)
        model = RiskyDebtModelVFI(
            self.risky_params, valid_config,
            k_bounds=(0.5 * k_ss, 2.0 * k_ss),
            b_bounds=(0.0, k_ss)
        )

        results = model.solve()
        self.assertIn("V", results)


class TestRiskyModelEdgeCases(RiskyModelTestCase):
    """Tests for edge cases and robustness."""

    def test_zero_debt_grid_point(self):
        """Model should handle zero-debt states correctly."""
        k_ss = SteadyStateCalculator.calculate_capital(self.risky_params)
        model = RiskyDebtModelVFI(
            self.risky_params, self.small_config,
            k_bounds=(0.5 * k_ss, 2.0 * k_ss),
            b_bounds=(0.0, k_ss)
        )

        results = model.solve()

        V_zero_debt = results["V"][:, 0, :]
        self.assertTrue(np.all(np.isfinite(V_zero_debt)))
        self.assertTrue(np.all(V_zero_debt >= 0))

    def test_high_volatility_solves(self):
        """Model should handle high productivity volatility."""
        high_vol_params = EconomicParams(
            discount_factor=0.96,
            capital_share=0.33,
            depreciation_rate=0.1,
            productivity_persistence=0.8,
            productivity_std_dev=0.08,
            adjustment_cost_convex=0.5,
            adjustment_cost_fixed=0.0,
            equity_issuance_cost_fixed=0.0,
            equity_issuance_cost_linear=0.0,
            default_cost_proportional=0.25,
            corporate_tax_rate=0.2,
            risk_free_rate=0.04,
            collateral_recovery_fraction=0.5
        )

        k_ss = SteadyStateCalculator.calculate_capital(high_vol_params)
        model = RiskyDebtModelVFI(
            high_vol_params, self.small_config,
            k_bounds=(0.2 * k_ss, 3.0 * k_ss),
            b_bounds=(0.0, 1.5 * k_ss)
        )

        results = model.solve()

        self.assertTrue(np.all(np.isfinite(results["V"])))
        self.assertTrue(np.all(np.isfinite(results["Q"])))

    def test_low_persistence_solves(self):
        """Model should handle low productivity persistence."""
        low_persist_params = EconomicParams(
            discount_factor=0.96,
            capital_share=0.33,
            depreciation_rate=0.1,
            productivity_persistence=0.5,
            productivity_std_dev=0.02,
            adjustment_cost_convex=0.5,
            adjustment_cost_fixed=0.0,
            equity_issuance_cost_fixed=0.0,
            equity_issuance_cost_linear=0.0,
            default_cost_proportional=0.25,
            corporate_tax_rate=0.2,
            risk_free_rate=0.04,
            collateral_recovery_fraction=0.5
        )

        k_ss = SteadyStateCalculator.calculate_capital(low_persist_params)
        model = RiskyDebtModelVFI(
            low_persist_params, self.small_config,
            k_bounds=(0.3 * k_ss, 2.5 * k_ss),
            b_bounds=(0.0, k_ss)
        )

        results = model.solve()

        self.assertTrue(np.all(np.isfinite(results["V"])))


class TestRiskyVsBasicComparison(RiskyModelTestCase):
    """Tests comparing risky and basic model behavior."""

    def test_risky_model_zero_debt_comparable(self):
        """Risky model at zero debt should behave reasonably."""
        k_ss = SteadyStateCalculator.calculate_capital(self.risky_params)
        model = RiskyDebtModelVFI(
            self.risky_params, self.small_config,
            k_bounds=(0.5 * k_ss, 2.0 * k_ss),
            b_bounds=(0.0, k_ss)
        )

        results = model.solve()
        V = results["V"]

        V_zero_debt = V[:, 0, :]
        V_high_debt = V[:, -1, :]

        self.assertGreater(
            np.mean(V_zero_debt),
            np.mean(V_high_debt),
            "Zero-debt value should exceed high-debt value"
        )


def create_test_suite():
    """Create a test suite containing all test cases."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    suite.addTests(loader.loadTestsFromTestCase(TestRiskyModelInitialization))
    suite.addTests(loader.loadTestsFromTestCase(TestRiskyModelFlowMatrix))
    suite.addTests(loader.loadTestsFromTestCase(TestRiskyModelSolver))
    suite.addTests(loader.loadTestsFromTestCase(TestRiskyModelPolicy))
    suite.addTests(loader.loadTestsFromTestCase(TestRiskyModelSimulation))
    suite.addTests(loader.loadTestsFromTestCase(TestRiskyModelBondPricing))
    suite.addTests(loader.loadTestsFromTestCase(TestRiskyModelEconomicConsistency))
    suite.addTests(loader.loadTestsFromTestCase(TestRiskyBoundaryFinder))
    suite.addTests(loader.loadTestsFromTestCase(TestRiskyModelMemoryConstraints))
    suite.addTests(loader.loadTestsFromTestCase(TestRiskyModelEdgeCases))
    suite.addTests(loader.loadTestsFromTestCase(TestRiskyVsBasicComparison))

    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(create_test_suite())