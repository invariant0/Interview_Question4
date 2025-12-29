"""Unit tests for Tauchen discretization method."""

import unittest
import tensorflow as tf
import numpy as np

from econ_models.core.grids.tauchen import tauchen_discretization


class TestTauchenDiscretization(unittest.TestCase):
    """Tests for the Tauchen AR(1) discretization."""

    def test_output_shapes(self):
        """Test that output shapes match requested grid size."""
        n = 7
        z, P = tauchen_discretization(n=n, rho=0.9, sigma=0.02)
        
        self.assertEqual(z.shape, (n,))
        self.assertEqual(P.shape, (n, n))

    def test_transition_matrix_rows_sum_to_one(self):
        """Test that each row of transition matrix sums to 1."""
        z, P = tauchen_discretization(n=11, rho=0.9, sigma=0.02)
        row_sums = tf.reduce_sum(P, axis=1)
        
        np.testing.assert_allclose(row_sums.numpy(), np.ones(11), rtol=1e-6)

    def test_transition_matrix_non_negative(self):
        """Test that all transition probabilities are non-negative."""
        z, P = tauchen_discretization(n=9, rho=0.95, sigma=0.01)
        
        self.assertTrue(tf.reduce_all(P >= 0).numpy())

    def test_grid_is_positive(self):
        """Test that z grid (in levels) is strictly positive."""
        z, P = tauchen_discretization(n=7, rho=0.9, sigma=0.05)
        
        self.assertTrue(tf.reduce_all(z > 0).numpy())

    def test_grid_symmetric_in_logs(self):
        """Test that log(z) grid is symmetric around zero."""
        z, P = tauchen_discretization(n=9, rho=0.9, sigma=0.02)
        log_z = tf.math.log(z).numpy()
        
        # Check symmetry: log_z[i] = -log_z[n-1-i]
        np.testing.assert_allclose(log_z, -log_z[::-1], atol=1e-6)

    def test_grid_contains_unity(self):
        """Test that z=1 (log z=0) is in the grid for odd n."""
        z, P = tauchen_discretization(n=9, rho=0.9, sigma=0.02)
        middle_idx = 4
        
        np.testing.assert_allclose(z[middle_idx].numpy(), 1.0, rtol=1e-6)

    def test_higher_persistence_concentrates_diagonal(self):
        """Test that higher rho puts more weight on diagonal."""
        _, P_low = tauchen_discretization(n=7, rho=0.5, sigma=0.02)
        _, P_high = tauchen_discretization(n=7, rho=0.99, sigma=0.02)
        
        diag_low = tf.linalg.diag_part(P_low)
        diag_high = tf.linalg.diag_part(P_high)
        
        # Higher persistence should have higher diagonal entries
        self.assertGreater(
            tf.reduce_mean(diag_high).numpy(),
            tf.reduce_mean(diag_low).numpy()
        )

    def test_zero_persistence_uniform_rows(self):
        """Test that rho=0 gives identical rows (iid process)."""
        z, P = tauchen_discretization(n=7, rho=0.0, sigma=0.1)
        
        # All rows should be identical for iid process
        for i in range(1, 7):
            np.testing.assert_allclose(
                P[i].numpy(), P[0].numpy(), rtol=1e-5
            )

    def test_larger_sigma_spreads_probability(self):
        """Test that larger sigma spreads transition probability."""
        _, P_small = tauchen_discretization(n=9, rho=0.9, sigma=0.01)
        _, P_large = tauchen_discretization(n=9, rho=0.9, sigma=0.10)
        
        # Larger sigma should have lower diagonal (more spread)
        diag_small = tf.linalg.diag_part(P_small)
        diag_large = tf.linalg.diag_part(P_large)
        
        self.assertGreater(
            tf.reduce_mean(diag_small).numpy(),
            tf.reduce_mean(diag_large).numpy()
        )


class TestTauchenEdgeCases(unittest.TestCase):
    """Edge case tests for Tauchen discretization."""

    def test_small_grid(self):
        """Test with minimum viable grid size."""
        z, P = tauchen_discretization(n=3, rho=0.9, sigma=0.02)
        
        self.assertEqual(z.shape, (3,))
        self.assertEqual(P.shape, (3, 3))

    def test_large_grid(self):
        """Test with larger grid size."""
        z, P = tauchen_discretization(n=51, rho=0.9, sigma=0.02)
        
        self.assertEqual(z.shape, (51,))
        np.testing.assert_allclose(
            tf.reduce_sum(P, axis=1).numpy(), 
            np.ones(51), 
            rtol=1e-6
        )

    def test_custom_width_parameter(self):
        """Test with custom m (width) parameter."""
        z_default, _ = tauchen_discretization(n=7, rho=0.9, sigma=0.02, m=3.0)
        z_wide, _ = tauchen_discretization(n=7, rho=0.9, sigma=0.02, m=5.0)
        
        # Wider m should give larger range
        self.assertGreater(z_wide[-1].numpy(), z_default[-1].numpy())
        self.assertLess(z_wide[0].numpy(), z_default[0].numpy())


if __name__ == '__main__':
    unittest.main()