"""Integration tests for Basic RBC Model."""

import unittest
import numpy as np
import tensorflow as tf

from econ_models.vfi.basic import BasicModelVFI
from econ_models.core.parameters import ModelParameters
from econ_models.core.config import ModelConfig


class TestBasicModelIntegration(unittest.TestCase):
    """Integration tests for the complete Basic RBC model."""

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
            productivity_volatility=0.02
        )
        
        cls.config = ModelConfig(
            n_capital=15,
            n_productivity=5,
            tol_vfi=1e-6,
            max_iter_vfi=500
        )

    def test_model_solves_without_error(self):
        """Test that model solves completely without raising exceptions."""
        model = BasicModelVFI(self.params, self.config)
        result = model.solve()
        
        self.assertIn('V', result)
        self.assertIn('K', result)
        self.assertIn('Z', result)

    def test_value_function_shape(self):
        """Test that value function has correct shape."""
        model = BasicModelVFI(self.params, self.config)
        result = model.solve()
        
        self.assertEqual(
            result['V'].shape,
            (self.config.n_capital, self.config.n_productivity)
        )

    def test_value_function_finite(self):
        """Test that value function contains only finite values."""
        model = BasicModelVFI(self.params, self.config)
        result = model.solve()
        
        self.assertTrue(np.all(np.isfinite(result['V'])))

    def test_value_function_positive(self):
        """Test that value function is positive (for this parameterization)."""
        model = BasicModelVFI(self.params, self.config)
        result = model.solve()
        
        self.assertTrue(np.all(result['V'] > 0))

    def test_value_increasing_in_capital(self):
        """Test that value is weakly increasing in capital for each z."""
        model = BasicModelVFI(self.params, self.config)
        result = model.solve()
        
        V = result['V']
        for z_idx in range(self.config.n_productivity):
            diffs = np.diff(V[:, z_idx])
            # Allow small numerical violations
            self.assertTrue(np.all(diffs >= -1e-6))

    def test_value_increasing_in_productivity(self):
        """Test that value is increasing in productivity for each k."""
        model = BasicModelVFI(self.params, self.config)
        result = model.solve()
        
        V = result['V']
        for k_idx in range(self.config.n_capital):
            diffs = np.diff(V[k_idx, :])
            self.assertTrue(np.all(diffs >= -1e-6))

    def test_grids_are_sorted(self):
        """Test that returned grids are monotonically increasing."""
        model = BasicModelVFI(self.params, self.config)
        result = model.solve()
        
        self.assertTrue(np.all(np.diff(result['K']) > 0))
        self.assertTrue(np.all(np.diff(result['Z']) > 0))

    def test_policy_indices_valid(self):
        """Test that policy indices are within grid bounds."""
        model = BasicModelVFI(self.params, self.config)
        result = model.solve()
        
        V_tensor = tf.constant(result['V'], dtype=tf.float64)
        policy_idx = model.get_policy_indices(V_tensor)
        
        self.assertTrue(np.all(policy_idx >= 0))
        self.assertTrue(np.all(policy_idx < self.config.n_capital))

    def test_simulation_runs(self):
        """Test that simulation completes without error."""
        model = BasicModelVFI(self.params, self.config)
        result = model.solve()
        
        stats = model.simulate(result['V'], t_steps=1000)
        
        self.assertIn('min_hit_pct', stats)
        self.assertIn('max_hit_pct', stats)
        self.assertTrue(0 <= stats['min_hit_pct'] <= 1)
        self.assertTrue(0 <= stats['max_hit_pct'] <= 1)

    def test_custom_k_bounds(self):
        """Test that custom capital bounds are respected."""
        custom_bounds = (0.3, 3.0)
        model = BasicModelVFI(self.params, self.config, k_bounds=custom_bounds)
        result = model.solve()
        
        K = result['K']
        self.assertGreaterEqual(K.min(), custom_bounds[0] - 1e-6)
        self.assertLessEqual(K.max(), custom_bounds[1] + 1e-6)

    def test_different_grid_sizes(self):
        """Test that model works with different grid sizes."""
        for n_k in [10, 20]:
            for n_z in [3, 7]:
                config = ModelConfig(
                    n_capital=n_k,
                    n_productivity=n_z,
                    tol_vfi=1e-5,
                    max_iter_vfi=300
                )
                model = BasicModelVFI(self.params, config)
                result = model.solve()
                
                self.assertEqual(result['V'].shape, (n_k, n_z))


class TestBasicModelEconomicProperties(unittest.TestCase):
    """Tests for economic properties of the solution."""

    def setUp(self):
        """Set up baseline model."""
        self.params = ModelParameters(
            discount_factor=0.96,
            depreciation_rate=0.1,
            capital_share=0.33,
            corporate_tax_rate=0.2,
            adjustment_cost_param=0.5,
            productivity_persistence=0.9,
            productivity_volatility=0.02
        )
        
        self.config = ModelConfig(
            n_capital=20,
            n_productivity=5,
            tol_vfi=1e-6,
            max_iter_vfi=500
        )

    def test_higher_beta_increases_value(self):
        """Test that higher discount factor increases value function."""
        params_low = ModelParameters(
            discount_factor=0.90,
            depreciation_rate=0.1,
            capital_share=0.33,
            corporate_tax_rate=0.2,
            adjustment_cost_param=0.5,
            productivity_persistence=0.9,
            productivity_volatility=0.02
        )
        
        params_high = ModelParameters(
            discount_factor=0.98,
            depreciation_rate=0.1,
            capital_share=0.33,
            corporate_tax_rate=0.2,
            adjustment_cost_param=0.5,
            productivity_persistence=0.9,
            productivity_volatility=0.02
        )
        
        model_low = BasicModelVFI(params_low, self.config)
        model_high = BasicModelVFI(params_high, self.config)
        
        result_low = model_low.solve()
        result_high = model_high.solve()
        
        # Higher beta should give higher values
        self.assertTrue(np.mean(result_high['V']) > np.mean(result_low['V']))

    def test_higher_tax_decreases_value(self):
        """Test that higher corporate tax decreases value function."""
        params_low_tax = ModelParameters(
            discount_factor=0.96,
            depreciation_rate=0.1,
            capital_share=0.33,
            corporate_tax_rate=0.1,
            adjustment_cost_param=0.5,
            productivity_persistence=0.9,
            productivity_volatility=0.02
        )
        
        params_high_tax = ModelParameters(
            discount_factor=0.96,
            depreciation_rate=0.1,
            capital_share=0.33,
            corporate_tax_rate=0.4,
            adjustment_cost_param=0.5,
            productivity_persistence=0.9,
            productivity_volatility=0.02
        )
        
        model_low = BasicModelVFI(params_low_tax, self.config)
        model_high = BasicModelVFI(params_high_tax, self.config)
        
        result_low = model_low.solve()
        result_high = model_high.solve()
        
        # Higher tax should give lower values
        self.assertTrue(np.mean(result_low['V']) > np.mean(result_high['V']))


if __name__ == '__main__':
    unittest.main()