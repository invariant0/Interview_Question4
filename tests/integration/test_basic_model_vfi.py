# tests/integration/test_basic_model_vfi.py
"""
Integration tests for the Basic RBC Model VFI implementation.

These tests verify the complete pipeline from parameter loading through
solving, policy extraction, and simulation using the unittest framework.
"""

import unittest
import numpy as np
import tensorflow as tf
from typing import Dict, Optional

from econ_models.config.economic_params import EconomicParams
from econ_models.config.vfi_config import GridConfig
from econ_models.vfi.basic import BasicModelVFI
from econ_models.vfi.bounds import BoundaryFinder
from econ_models.econ import SteadyStateCalculator


class BasicModelTestCase(unittest.TestCase):
    """Base test case with common setup for basic model tests."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures used by all test methods."""
        cls.basic_params = EconomicParams(
            discount_factor=0.96,
            capital_share=0.33,
            depreciation_rate=0.1,
            productivity_persistence=0.9,
            productivity_std_dev=0.02,
            adjustment_cost_convex=0.5,
            adjustment_cost_fixed=0.0,
            equity_issuance_cost_fixed=0.0,
            equity_issuance_cost_linear=0.0,
            default_cost_proportional=0.0,
            corporate_tax_rate=0.2,
            risk_free_rate=0.04,
            collateral_recovery_fraction=None
        )

        cls.small_config = GridConfig(
            n_capital=25,
            n_productivity=5,
            n_debt=0,
            tauchen_width=2.5,
            tol_vfi=1e-6,
            max_iter_vfi=500
        )

        cls.medium_config = GridConfig(
            n_capital=50,
            n_productivity=7,
            n_debt=0,
            tauchen_width=3.0,
            tol_vfi=1e-7,
            max_iter_vfi=1000
        )

    def _create_model_with_bounds(
        self,
        params: Optional[EconomicParams] = None,
        config: Optional[GridConfig] = None
    ) -> tuple:
        """Helper to create model with computed bounds."""
        params = params or self.basic_params
        config = config or self.small_config

        k_ss = SteadyStateCalculator.calculate_capital(params)
        k_bounds = (0.3 * k_ss, 3.0 * k_ss)

        model = BasicModelVFI(params, config, k_bounds=k_bounds)
        return model, k_bounds, k_ss


class TestBasicModelInitialization(BasicModelTestCase):
    """Tests for model initialization and grid construction."""

    def test_model_initializes_with_valid_params(self):
        """Model should initialize without errors given valid parameters."""
        model = BasicModelVFI(self.basic_params, self.small_config)

        self.assertEqual(model.params, self.basic_params)
        self.assertEqual(model.config, self.small_config)
        self.assertIsNotNone(model.k_grid)
        self.assertIsNotNone(model.z_grid)
        self.assertIsNotNone(model.P)

    def test_grid_dimensions_match_config(self):
        """Grid dimensions should match configuration."""
        model = BasicModelVFI(self.basic_params, self.small_config)

        self.assertEqual(
            model.k_grid.shape[0],
            self.small_config.n_capital
        )
        self.assertEqual(
            model.z_grid.shape[0],
            self.small_config.n_productivity
        )
        self.assertEqual(
            model.P.shape,
            (self.small_config.n_productivity, self.small_config.n_productivity)
        )

    def test_capital_grid_is_monotonic(self):
        """Capital grid should be strictly increasing."""
        model = BasicModelVFI(self.basic_params, self.small_config)
        k_vals = model.k_grid.numpy()

        diffs = np.diff(k_vals)
        self.assertTrue(
            np.all(diffs > 0),
            "Capital grid must be strictly monotonic"
        )

    def test_productivity_grid_centered_around_one(self):
        """Productivity grid (in levels) should be approximately centered around 1."""
        model = BasicModelVFI(self.basic_params, self.small_config)
        z_vals = model.z_grid.numpy()

        # For Tauchen method with AR(1) in logs, the levels are exp(log_z)
        # The mean in levels should be close to 1 (or slightly above due to Jensen's inequality)
        mean_z = np.mean(z_vals)
        self.assertGreater(mean_z, 0.5, f"Z grid mean {mean_z} should be positive")
        self.assertLess(mean_z, 2.0, f"Z grid mean {mean_z} should be reasonable")

    def test_productivity_grid_is_monotonic(self):
        """Productivity grid should be strictly increasing."""
        model = BasicModelVFI(self.basic_params, self.small_config)
        z_vals = model.z_grid.numpy()

        diffs = np.diff(z_vals)
        self.assertTrue(
            np.all(diffs > 0),
            "Productivity grid must be strictly monotonic"
        )

    def test_transition_matrix_is_stochastic(self):
        """Transition matrix rows should sum to 1."""
        model = BasicModelVFI(self.basic_params, self.small_config)
        P = model.P.numpy()

        row_sums = np.sum(P, axis=1)
        np.testing.assert_allclose(
            row_sums, 1.0, rtol=1e-5,
            err_msg="Transition matrix rows must sum to 1"
        )

    def test_custom_bounds_respected(self):
        """Custom capital bounds should be applied correctly."""
        custom_bounds = (0.5, 5.0)
        model = BasicModelVFI(
            self.basic_params, self.small_config, k_bounds=custom_bounds
        )
        k_vals = model.k_grid.numpy()

        self.assertGreaterEqual(k_vals[0], custom_bounds[0] - 1e-6)
        self.assertLessEqual(k_vals[-1], custom_bounds[1] + 1e-6)

    def test_steady_state_capital_computed(self):
        """Model should compute steady state capital."""
        model, _, k_ss = self._create_model_with_bounds()

        self.assertGreater(k_ss, 0)
        self.assertIsNotNone(model.k_ss)


class TestBasicModelFlowMatrix(BasicModelTestCase):
    """Tests for flow matrix computation."""

    def test_flow_matrix_shape(self):
        """Flow matrix should have correct shape."""
        model = BasicModelVFI(self.basic_params, self.small_config)
        flow = model.compute_flow()

        expected_shape = (
            self.small_config.n_capital,
            self.small_config.n_capital,
            self.small_config.n_productivity
        )
        self.assertEqual(flow.shape, expected_shape)

    def test_flow_matrix_finite(self):
        """Flow matrix should not contain NaN values."""
        model = BasicModelVFI(self.basic_params, self.small_config)
        flow = model.compute_flow().numpy()

        self.assertFalse(
            np.any(np.isnan(flow)),
            "Flow matrix contains NaN values"
        )

    def test_flow_increases_with_productivity(self):
        """Higher productivity should yield higher flow values."""
        model = BasicModelVFI(self.basic_params, self.small_config)
        flow = model.compute_flow().numpy()

        mid_k = self.small_config.n_capital // 2
        flow_by_z = flow[mid_k, mid_k, :]

        diffs = np.diff(flow_by_z)
        self.assertTrue(
            np.all(diffs >= -1e-10),
            "Flow should weakly increase with productivity"
        )

    def test_flow_dtype_is_float(self):
        """Flow matrix should be a float tensor."""
        model = BasicModelVFI(self.basic_params, self.small_config)
        flow = model.compute_flow()

        # Accept either float32 or float64
        self.assertTrue(
            flow.dtype in [tf.float32, tf.float64],
            f"Flow dtype {flow.dtype} should be float32 or float64"
        )


class TestBasicModelSolver(BasicModelTestCase):
    """Tests for the VFI solving process."""

    @classmethod
    def setUpClass(cls):
        """Set up solved model for solver tests."""
        super().setUpClass()

        k_ss = SteadyStateCalculator.calculate_capital(cls.basic_params)
        k_bounds = (0.3 * k_ss, 3.0 * k_ss)

        cls.model = BasicModelVFI(
            cls.basic_params, cls.small_config, k_bounds=k_bounds
        )
        cls.results = cls.model.solve()

    def test_solve_returns_required_keys(self):
        """Solution dictionary should contain all required arrays."""
        self.assertIn("V", self.results)
        self.assertIn("K", self.results)
        self.assertIn("Z", self.results)

    def test_value_function_shape(self):
        """Value function should have correct shape."""
        expected_shape = (
            self.small_config.n_capital,
            self.small_config.n_productivity
        )
        self.assertEqual(self.results["V"].shape, expected_shape)

    def test_value_function_is_nonnegative(self):
        """Value function should be non-negative everywhere."""
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

    def test_value_function_increases_with_capital(self):
        """Value function should increase with capital for fixed productivity."""
        V = self.results["V"]

        for z_idx in range(V.shape[1]):
            v_slice = V[:, z_idx]
            violations = np.sum(np.diff(v_slice) < -1e-6)
            self.assertLessEqual(
                violations, 2,
                f"V should be increasing in K for z_idx={z_idx}"
            )

    def test_value_function_increases_with_productivity(self):
        """Value function should increase with productivity for fixed capital."""
        V = self.results["V"]

        mid_k = V.shape[0] // 2
        v_slice = V[mid_k, :]

        diffs = np.diff(v_slice)
        self.assertTrue(
            np.all(diffs >= -1e-6),
            "V should be increasing in Z"
        )

    def test_convergence_deterministic(self):
        """VFI should converge to same solution on repeated runs."""
        results2 = self.model.solve()

        np.testing.assert_allclose(
            self.results["V"], results2["V"], rtol=1e-5,
            err_msg="VFI should be deterministic"
        )

    def test_grids_match_config(self):
        """Returned grids should match configuration."""
        self.assertEqual(
            len(self.results["K"]),
            self.small_config.n_capital
        )
        self.assertEqual(
            len(self.results["Z"]),
            self.small_config.n_productivity
        )


class TestBasicModelPolicy(BasicModelTestCase):
    """Tests for policy function extraction."""

    @classmethod
    def setUpClass(cls):
        """Set up solved model for policy tests."""
        super().setUpClass()

        k_ss = SteadyStateCalculator.calculate_capital(cls.basic_params)
        k_bounds = (0.3 * k_ss, 3.0 * k_ss)

        cls.model = BasicModelVFI(
            cls.basic_params, cls.small_config, k_bounds=k_bounds
        )
        cls.results = cls.model.solve()
        
        # Use the same dtype as the transition matrix P
        p_dtype = cls.model.P.dtype
        cls.v_tensor = tf.constant(cls.results["V"], dtype=p_dtype)
        cls.policy = cls.model.get_policy_indices(cls.v_tensor)

    def test_policy_indices_shape(self):
        """Policy indices should have correct shape."""
        expected_shape = (
            self.small_config.n_capital,
            self.small_config.n_productivity
        )
        self.assertEqual(self.policy.shape, expected_shape)

    def test_policy_indices_in_valid_range(self):
        """Policy indices should be valid grid indices."""
        policy_np = self.policy.numpy()

        self.assertTrue(np.all(policy_np >= 0))
        self.assertTrue(np.all(policy_np < self.small_config.n_capital))

    def test_policy_is_continuous_ish(self):
        """Policy should not have extreme jumps between adjacent states."""
        policy_np = self.policy.numpy()

        max_jump = np.max(np.abs(np.diff(policy_np, axis=0)))
        max_allowed = int(0.3 * self.small_config.n_capital)

        self.assertLessEqual(
            max_jump, max_allowed,
            f"Policy has discontinuous jump of {max_jump} indices"
        )

    def test_policy_dtype(self):
        """Policy indices should be integer type."""
        self.assertEqual(self.policy.dtype, tf.int32)


class TestBasicModelSimulation(BasicModelTestCase):
    """Tests for model simulation."""

    @classmethod
    def setUpClass(cls):
        """Set up solved model for simulation tests."""
        super().setUpClass()

        k_ss = SteadyStateCalculator.calculate_capital(cls.basic_params)
        k_bounds = (0.3 * k_ss, 3.0 * k_ss)

        cls.model = BasicModelVFI(
            cls.basic_params, cls.small_config, k_bounds=k_bounds
        )
        cls.results = cls.model.solve()

    def test_simulation_returns_history_and_stats(self):
        """Simulation should return history object and statistics dict."""
        history, stats = self.model.simulate(
            self.results["V"], n_steps=100, n_batches=50, seed=42
        )

        self.assertIsNotNone(history)
        self.assertIsInstance(stats, dict)
        self.assertIn("min_hit_pct", stats)
        self.assertIn("max_hit_pct", stats)
        self.assertIn("total_observations", stats)

    def test_simulation_history_dimensions(self):
        """Simulation history should have correct dimensions."""
        n_steps, n_batches = 100, 50

        history, _ = self.model.simulate(
            self.results["V"], n_steps=n_steps, n_batches=n_batches, seed=42
        )

        self.assertEqual(history.n_steps, n_steps)
        self.assertEqual(history.n_batches, n_batches)
        self.assertEqual(history.total_observations, n_steps * n_batches)
        self.assertIn("k_idx", history.trajectories)
        self.assertIn("z_idx", history.trajectories)

    def test_simulation_trajectory_shapes(self):
        """Trajectory arrays should have correct shapes."""
        n_steps, n_batches = 100, 50

        history, _ = self.model.simulate(
            self.results["V"], n_steps=n_steps, n_batches=n_batches, seed=42
        )

        self.assertEqual(
            history.trajectories["k_idx"].shape,
            (n_batches, n_steps)
        )
        self.assertEqual(
            history.trajectories["z_idx"].shape,
            (n_batches, n_steps)
        )

    def test_simulation_trajectories_in_bounds(self):
        """Simulated trajectories should stay within grid bounds."""
        history, _ = self.model.simulate(
            self.results["V"], n_steps=200, n_batches=100, seed=42
        )

        k_traj = history.trajectories["k_idx"]
        z_traj = history.trajectories["z_idx"]

        self.assertTrue(np.all(k_traj >= 0))
        self.assertTrue(np.all(k_traj < self.small_config.n_capital))
        self.assertTrue(np.all(z_traj >= 0))
        self.assertTrue(np.all(z_traj < self.small_config.n_productivity))

    def test_simulation_reproducibility(self):
        """Simulation with same seed should produce identical results."""
        history1, stats1 = self.model.simulate(
            self.results["V"], n_steps=100, n_batches=50, seed=12345
        )
        history2, stats2 = self.model.simulate(
            self.results["V"], n_steps=100, n_batches=50, seed=12345
        )

        np.testing.assert_array_equal(
            history1.trajectories["k_idx"],
            history2.trajectories["k_idx"]
        )
        self.assertEqual(stats1, stats2)

    def test_boundary_hit_rates_reasonable(self):
        """Boundary hit rates should be reasonable with proper bounds."""
        _, stats = self.model.simulate(
            self.results["V"], n_steps=500, n_batches=200, seed=42
        )

        self.assertLess(
            stats["min_hit_pct"], 0.2,
            "Too many lower bound hits"
        )
        self.assertLess(
            stats["max_hit_pct"], 0.2,
            "Too many upper bound hits"
        )

    def test_different_seeds_produce_different_results(self):
        """Different seeds should produce different trajectories."""
        history1, _ = self.model.simulate(
            self.results["V"], n_steps=100, n_batches=50, seed=111
        )
        history2, _ = self.model.simulate(
            self.results["V"], n_steps=100, n_batches=50, seed=222
        )

        self.assertFalse(
            np.array_equal(
                history1.trajectories["k_idx"],
                history2.trajectories["k_idx"]
            ),
            "Different seeds should produce different results"
        )


class TestBasicModelEconomicConsistency(BasicModelTestCase):
    """Tests for economic consistency of model behavior."""

    def test_higher_discount_increases_value(self):
        """Higher discount factor should increase value function."""
        low_beta_params = EconomicParams(
            discount_factor=0.94,
            capital_share=0.33,
            depreciation_rate=0.1,
            productivity_persistence=0.9,
            productivity_std_dev=0.02,
            adjustment_cost_convex=0.5,
            adjustment_cost_fixed=0.0,
            equity_issuance_cost_fixed=0.0,
            equity_issuance_cost_linear=0.0,
            default_cost_proportional=0.0,
            corporate_tax_rate=0.2,
            risk_free_rate=0.04
        )

        high_beta_params = EconomicParams(
            discount_factor=0.98,
            capital_share=0.33,
            depreciation_rate=0.1,
            productivity_persistence=0.9,
            productivity_std_dev=0.02,
            adjustment_cost_convex=0.5,
            adjustment_cost_fixed=0.0,
            equity_issuance_cost_fixed=0.0,
            equity_issuance_cost_linear=0.0,
            default_cost_proportional=0.0,
            corporate_tax_rate=0.2,
            risk_free_rate=0.04
        )

        model_low = BasicModelVFI(low_beta_params, self.small_config)
        model_high = BasicModelVFI(high_beta_params, self.small_config)

        results_low = model_low.solve()
        results_high = model_high.solve()

        self.assertGreater(
            np.mean(results_high["V"]),
            np.mean(results_low["V"]),
            "Higher beta should yield higher value"
        )

    def test_steady_state_capital_in_grid(self):
        """Steady state capital should be within the grid."""
        k_ss = SteadyStateCalculator.calculate_capital(self.basic_params)
        model = BasicModelVFI(self.basic_params, self.small_config)

        k_min = model.k_grid.numpy()[0]
        k_max = model.k_grid.numpy()[-1]

        self.assertGreaterEqual(k_ss, k_min, "Steady state below grid minimum")
        self.assertLessEqual(k_ss, k_max, "Steady state above grid maximum")

    def test_lower_tax_increases_value(self):
        """Lower corporate tax rate should increase value function."""
        high_tax_params = EconomicParams(
            discount_factor=0.96,
            capital_share=0.33,
            depreciation_rate=0.1,
            productivity_persistence=0.9,
            productivity_std_dev=0.02,
            adjustment_cost_convex=0.5,
            adjustment_cost_fixed=0.0,
            equity_issuance_cost_fixed=0.0,
            equity_issuance_cost_linear=0.0,
            default_cost_proportional=0.0,
            corporate_tax_rate=0.4,
            risk_free_rate=0.04
        )

        low_tax_params = EconomicParams(
            discount_factor=0.96,
            capital_share=0.33,
            depreciation_rate=0.1,
            productivity_persistence=0.9,
            productivity_std_dev=0.02,
            adjustment_cost_convex=0.5,
            adjustment_cost_fixed=0.0,
            equity_issuance_cost_fixed=0.0,
            equity_issuance_cost_linear=0.0,
            default_cost_proportional=0.0,
            corporate_tax_rate=0.1,
            risk_free_rate=0.04
        )

        model_high = BasicModelVFI(high_tax_params, self.small_config)
        model_low = BasicModelVFI(low_tax_params, self.small_config)

        results_high = model_high.solve()
        results_low = model_low.solve()

        self.assertGreater(
            np.mean(results_low["V"]),
            np.mean(results_high["V"]),
            "Lower tax should yield higher value"
        )


class TestBasicBoundaryFinder(BasicModelTestCase):
    """Tests for automatic boundary discovery."""

    def test_boundary_finder_returns_valid_bounds(self):
        """Boundary finder should return valid bound dictionaries."""
        finder = BoundaryFinder(
            self.basic_params, self.small_config,
            threshold=0.05,
            n_steps=50,
            n_batches=50,
            seed=42
        )
        bounds = finder.find_basic_bounds()

        self.assertIn("k_bounds_original", bounds)
        self.assertIn("z_bounds_original", bounds)
        self.assertIn("k_bounds_add_margin", bounds)
        self.assertIn("z_bounds_add_margin", bounds)
        self.assertIn("vfi_solution", bounds)

    def test_margin_bounds_wider_than_original(self):
        """Margin-adjusted bounds should be wider than original."""
        finder = BoundaryFinder(
            self.basic_params, self.small_config,
            threshold=0.05,
            margin=1.1,
            n_steps=50,
            n_batches=50,
            seed=42
        )
        bounds = finder.find_basic_bounds()

        k_orig = bounds["k_bounds_original"]
        k_margin = bounds["k_bounds_add_margin"]

        self.assertLess(
            k_margin[0], k_orig[0],
            "Lower margin bound should be smaller"
        )
        self.assertGreater(
            k_margin[1], k_orig[1],
            "Upper margin bound should be larger"
        )

    def test_boundary_finder_with_custom_threshold(self):
        """Boundary finder should respect custom threshold."""
        finder = BoundaryFinder(
            self.basic_params, self.small_config,
            threshold=0.02,
            n_steps=50,
            n_batches=50,
            seed=42
        )

        self.assertEqual(finder.threshold, 0.02)


class TestBasicModelEdgeCases(BasicModelTestCase):
    """Tests for edge cases and error handling."""

    def test_invalid_discount_factor_raises(self):
        """Invalid discount factor should raise ValueError."""
        with self.assertRaises(ValueError) as context:
            EconomicParams(
                discount_factor=1.5,
                capital_share=0.33,
                depreciation_rate=0.1,
                productivity_persistence=0.9,
                productivity_std_dev=0.02,
                adjustment_cost_convex=0.5,
                adjustment_cost_fixed=0.0,
                equity_issuance_cost_fixed=0.0,
                equity_issuance_cost_linear=0.0,
                default_cost_proportional=0.0,
                corporate_tax_rate=0.2,
                risk_free_rate=0.04
            )

        self.assertIn("discount", str(context.exception).lower())

    def test_zero_discount_factor_raises(self):
        """Zero discount factor should raise ValueError."""
        with self.assertRaises(ValueError):
            EconomicParams(
                discount_factor=0.0,
                capital_share=0.33,
                depreciation_rate=0.1,
                productivity_persistence=0.9,
                productivity_std_dev=0.02,
                adjustment_cost_convex=0.5,
                adjustment_cost_fixed=0.0,
                equity_issuance_cost_fixed=0.0,
                equity_issuance_cost_linear=0.0,
                default_cost_proportional=0.0,
                corporate_tax_rate=0.2,
                risk_free_rate=0.04
            )

    def test_minimal_grid_solves(self):
        """Model should solve even with minimal grid."""
        minimal_config = GridConfig(
            n_capital=10,
            n_productivity=3,
            n_debt=0,
            tauchen_width=2.0,
            tol_vfi=1e-5,
            max_iter_vfi=200
        )

        model = BasicModelVFI(self.basic_params, minimal_config)
        results = model.solve()

        self.assertEqual(results["V"].shape, (10, 3))
        self.assertTrue(np.all(np.isfinite(results["V"])))

    def test_negative_discount_factor_raises(self):
        """Negative discount factor should raise ValueError."""
        with self.assertRaises(ValueError):
            EconomicParams(
                discount_factor=-0.5,
                capital_share=0.33,
                depreciation_rate=0.1,
                productivity_persistence=0.9,
                productivity_std_dev=0.02,
                adjustment_cost_convex=0.5,
                adjustment_cost_fixed=0.0,
                equity_issuance_cost_fixed=0.0,
                equity_issuance_cost_linear=0.0,
                default_cost_proportional=0.0,
                corporate_tax_rate=0.2,
                risk_free_rate=0.04
            )


class TestBasicModelGridBuilder(BasicModelTestCase):
    """Tests for grid building functionality."""

    def test_log_spaced_capital_grid(self):
        """Capital grid should be log-spaced by default."""
        model, _, _ = self._create_model_with_bounds()
        k_vals = model.k_grid.numpy()

        log_k = np.log(k_vals)
        log_diffs = np.diff(log_k)

        std_diff = np.std(log_diffs)
        self.assertLess(
            std_diff, 0.01,
            "Log-spaced grid should have uniform log differences"
        )

    def test_productivity_grid_uses_tauchen(self):
        """Productivity grid should be built using Tauchen method."""
        model = BasicModelVFI(self.basic_params, self.small_config)

        self.assertEqual(
            len(model.z_grid.numpy()),
            self.small_config.n_productivity
        )
        self.assertIsNotNone(model.P)


def create_test_suite():
    """Create a test suite containing all test cases."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    suite.addTests(loader.loadTestsFromTestCase(TestBasicModelInitialization))
    suite.addTests(loader.loadTestsFromTestCase(TestBasicModelFlowMatrix))
    suite.addTests(loader.loadTestsFromTestCase(TestBasicModelSolver))
    suite.addTests(loader.loadTestsFromTestCase(TestBasicModelPolicy))
    suite.addTests(loader.loadTestsFromTestCase(TestBasicModelSimulation))
    suite.addTests(loader.loadTestsFromTestCase(TestBasicModelEconomicConsistency))
    suite.addTests(loader.loadTestsFromTestCase(TestBasicBoundaryFinder))
    suite.addTests(loader.loadTestsFromTestCase(TestBasicModelEdgeCases))
    suite.addTests(loader.loadTestsFromTestCase(TestBasicModelGridBuilder))

    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(create_test_suite())