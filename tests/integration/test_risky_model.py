"""Integration tests for Risky Debt Model."""

import unittest
import numpy as np
import tensorflow as tf

from econ_models.vfi.risky_debt import RiskyDebtModelVFI
from econ_models.core.parameters import ModelParameters
from econ_models.core.config import ModelConfig


class TestRiskyDebtModelIntegration(unittest.TestCase):
    """Integration tests for the complete Risky Debt model."""

    @classmethod
    def setUpClass(cls):
        """Set up model parameters and config once for all tests."""
        cls.params = ModelParameters(
            discount_factor=0.96,
            depreciation_rate=0.1,
            capital_share=0.33,
            corporate_tax_rate=0.2,
            adjustment_cost_param=0.5,
            productivity_persistence=0.9,
            productivity_volatility=0.02,
            risk_free_rate=0.04,
            collateral_recovery_fraction=0.5
        )
        
        # Small grids for faster testing
        cls.config = ModelConfig(
            n_capital=10,
            n_debt=8,
            n_productivity=3,
            tol_vfi=1e-5,
            max_iter_vfi=200,
            max_outer=5,
            tol_outer=1e-3
        )

    def test_model_solves_without_error(self):
        """Test that model solves completely without raising exceptions."""
        model = RiskyDebtModelVFI(self.params, self.config)
        result = model.solve()
        
        self.assertIn('V', result)
        self.assertIn('Q', result)
        self.assertIn('K', result)
        self.assertIn('B', result)
        self.assertIn('Z', result)

    def test_value_function_shape(self):
        """Test that value function has correct 3D shape."""
        model = RiskyDebtModelVFI(self.params, self.config)
        result = model.solve()
        
        expected_shape = (
            self.config.n_capital,
            self.config.n_debt,
            self.config.n_productivity
        )
        self.assertEqual(result['V'].shape, expected_shape)

    def test_bond_prices_shape(self):
        """Test that bond prices have correct 3D shape."""
        model = RiskyDebtModelVFI(self.params, self.config)
        result = model.solve()
        
        expected_shape = (
            self.config.n_capital,
            self.config.n_debt,
            self.config.n_productivity
        )
        self.assertEqual(result['Q'].shape, expected_shape)

    def test_value_function_finite(self):
        """Test that value function contains only finite values."""
        model = RiskyDebtModelVFI(self.params, self.config)
        result = model.solve()
        
        self.assertTrue(np.all(np.isfinite(result['V'])))

    def test_bond_prices_bounded(self):
        """Test that bond prices are in valid range [0, 1/(1+r)]."""
        model = RiskyDebtModelVFI(self.params, self.config)
        result = model.solve()
        
        Q = result['Q']
        max_price = 1.0 / (1.0 + self.params.risk_free_rate)
        
        self.assertTrue(np.all(Q >= 0))
        self.assertTrue(np.all(Q <= max_price + 1e-6))

    def test_grids_sorted(self):
        """Test that all grids are monotonically increasing."""
        model = RiskyDebtModelVFI(self.params, self.config)
        result = model.solve()
        
        self.assertTrue(np.all(np.diff(result['K']) > 0))
        self.assertTrue(np.all(np.diff(result['B']) > 0))
        self.assertTrue(np.all(np.diff(result['Z']) > 0))

    def test_policy_indices_valid(self):
        """Test that policy indices are within grid bounds."""
        model = RiskyDebtModelVFI(self.params, self.config)
        result = model.solve()
        
        V_tensor = tf.constant(result['V'], dtype=tf.float64)
        pol_k, pol_b = model.get_policy_indices(V_tensor)
        
        self.assertTrue(np.all(pol_k >= 0))
        self.assertTrue(np.all(pol_k < self.config.n_capital))
        self.assertTrue(np.all(pol_b >= 0))
        self.assertTrue(np.all(pol_b < self.config.n_debt))

    def test_value_increasing_in_capital(self):
        """Test that value is weakly increasing in capital."""
        model = RiskyDebtModelVFI(self.params, self.config)
        result = model.solve()
        
        V = result['V']
        for b_idx in range(self.config.n_debt):
            for z_idx in range(self.config.n_productivity):
                diffs = np.diff(V[:, b_idx, z_idx])
                # Allow small numerical violations
                self.assertTrue(np.all(diffs >= -1e-4))

    def test_value_decreasing_in_debt(self):
        """Test that value is weakly decreasing in debt."""
        model = RiskyDebtModelVFI(self.params, self.config)
        result = model.solve()
        
        V = result['V']
        for k_idx in range(self.config.n_capital):
            for z_idx in range(self.config.n_productivity):
                diffs = np.diff(V[k_idx, :, z_idx])
                # Value should decrease as debt increases
                # Allow small numerical violations
                self.assertTrue(np.all(diffs <= 1e-4))

    def test_bond_price_decreasing_in_debt(self):
        """Test that bond prices decrease with more debt (higher default risk)."""
        model = RiskyDebtModelVFI(self.params, self.config)
        result = model.solve()
        
        Q = result['Q']
        # Average across states - higher debt should mean lower prices
        avg_Q_by_debt = np.mean(Q, axis=(0, 2))
        diffs = np.diff(avg_Q_by_debt)
        
        # Most differences should be negative or near zero
        self.assertTrue(np.mean(diffs) <= 0.01)


class TestRiskyDebtValidation(unittest.TestCase):
    """Tests for input validation."""

    def test_raises_on_n_debt_less_than_2(self):
        """Test that n_debt < 2 raises ValueError."""
        params = ModelParameters(
            discount_factor=0.96,
            risk_free_rate=0.04
        )
        
        config = ModelConfig(
            n_capital=10,
            n_debt=1,  # Invalid
            n_productivity=3
        )
        
        with self.assertRaises(ValueError):
            RiskyDebtModelVFI(params, config)

    def test_raises_on_large_grids(self):
        """Test that excessively large grids raise ValueError."""
        params = ModelParameters(
            discount_factor=0.96,
            risk_free_rate=0.04
        )
        
        config = ModelConfig(
            n_capital=150,  # Too large
            n_debt=50,
            n_productivity=5
        )
        
        model = RiskyDebtModelVFI(params, config)
        
        with self.assertRaises(ValueError):
            model.solve()


class TestRiskyDebtEconomicProperties(unittest.TestCase):
    """Tests for economic properties of the risky debt solution."""

    def setUp(self):
        """Set up baseline configuration."""
        self.config = ModelConfig(
            n_capital=12,
            n_debt=10,
            n_productivity=3,
            tol_vfi=1e-5,
            max_iter_vfi=200,
            max_outer=5,
            tol_outer=1e-3
        )

    def test_higher_recovery_increases_prices(self):
        """Test that higher collateral recovery increases bond prices."""
        params_low = ModelParameters(
            discount_factor=0.96,
            depreciation_rate=0.1,
            capital_share=0.33,
            corporate_tax_rate=0.2,
            adjustment_cost_param=0.5,
            productivity_persistence=0.9,
            productivity_volatility=0.02,
            risk_free_rate=0.04,
            collateral_recovery_fraction=0.3
        )
        
        params_high = ModelParameters(
            discount_factor=0.96,
            depreciation_rate=0.1,
            capital_share=0.33,
            corporate_tax_rate=0.2,
            adjustment_cost_param=0.5,
            productivity_persistence=0.9,
            productivity_volatility=0.02,
            risk_free_rate=0.04,
            collateral_recovery_fraction=0.7
        )
        
        model_low = RiskyDebtModelVFI(params_low, self.config)
        model_high = RiskyDebtModelVFI(params_high, self.config)
        
        result_low = model_low.solve()
        result_high = model_high.solve()
        
        # Higher recovery fraction should give higher bond prices on average
        self.assertTrue(np.mean(result_high['Q']) >= np.mean(result_low['Q']))

    def test_lower_volatility_increases_prices(self):
        """Test that lower productivity volatility increases bond prices."""
        params_high_vol = ModelParameters(
            discount_factor=0.96,
            depreciation_rate=0.1,
            capital_share=0.33,
            corporate_tax_rate=0.2,
            adjustment_cost_param=0.5,
            productivity_persistence=0.9,
            productivity_volatility=0.05,  # High volatility
            risk_free_rate=0.04,
            collateral_recovery_fraction=0.5
        )
        
        params_low_vol = ModelParameters(
            discount_factor=0.96,
            depreciation_rate=0.1,
            capital_share=0.33,
            corporate_tax_rate=0.2,
            adjustment_cost_param=0.5,
            productivity_persistence=0.9,
            productivity_volatility=0.01,  # Low volatility
            risk_free_rate=0.04,
            collateral_recovery_fraction=0.5
        )
        
        model_high = RiskyDebtModelVFI(params_high_vol, self.config)
        model_low = RiskyDebtModelVFI(params_low_vol, self.config)
        
        result_high = model_high.solve()
        result_low = model_low.solve()
        
        # Lower volatility = less default risk = higher prices
        self.assertTrue(np.mean(result_low['Q']) >= np.mean(result_high['Q']))


class TestRiskyDebtSimulation(unittest.TestCase):
    """Tests for simulation functionality."""

    @classmethod
    def setUpClass(cls):
        """Set up and solve model once."""
        cls.params = ModelParameters(
            discount_factor=0.96,
            depreciation_rate=0.1,
            capital_share=0.33,
            corporate_tax_rate=0.2,
            adjustment_cost_param=0.5,
            productivity_persistence=0.9,
            productivity_volatility=0.02,
            risk_free_rate=0.04,
            collateral_recovery_fraction=0.5
        )
        
        cls.config = ModelConfig(
            n_capital=10,
            n_debt=8,
            n_productivity=3,
            tol_vfi=1e-5,
            max_iter_vfi=200,
            max_outer=3,
            tol_outer=1e-3
        )
        
        cls.model = RiskyDebtModelVFI(cls.params, cls.config)
        cls.result = cls.model.solve()

    def test_simulation_runs(self):
        """Test that simulation completes without error."""
        stats = self.model.simulate(self.result['V'], t_steps=500)
        
        self.assertIn('k_min', stats)
        self.assertIn('k_max', stats)
        self.assertIn('b_min', stats)
        self.assertIn('b_max', stats)

    def test_simulation_stats_in_range(self):
        """Test that boundary hit percentages are in [0, 1]."""
        stats = self.model.simulate(self.result['V'], t_steps=500)
        
        for key in ['k_min', 'k_max', 'b_min', 'b_max']:
            self.assertTrue(0 <= stats[key] <= 1)


if __name__ == '__main__':
    unittest.main()