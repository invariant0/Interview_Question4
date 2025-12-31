"""Unit tests for Tauchen discretization method."""

import unittest
import tensorflow as tf
import numpy as np

from econ_models.grids.tauchen import tauchen_discretization


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
        
        # z = exp(x) where x is the log grid, so z is always positive
        self.assertTrue(tf.reduce_all(z > 0).numpy())

    def test_grid_symmetric_in_logs(self):
        """Test that log(z) grid is symmetric around zero."""
        z, P = tauchen_discretization(n=9, rho=0.9, sigma=0.02)
        log_z = tf.math.log(z).numpy()
        
        # Check symmetry: log_z[i] = -log_z[n-1-i]
        np.testing.assert_allclose(log_z, -log_z[::-1], atol=1e-6)

    def test_grid_contains_unity(self):
        """Test that z=1 (log z=0) is in the grid for odd n."""
        n = 9
        z, P = tauchen_discretization(n=n, rho=0.9, sigma=0.02)
        middle_idx = n // 2
        
        np.testing.assert_allclose(z[middle_idx].numpy(), 1.0, rtol=1e-6)

    def test_grid_is_monotonically_increasing(self):
        """Test that the z grid is strictly increasing."""
        z, P = tauchen_discretization(n=11, rho=0.9, sigma=0.02)
        
        diffs = z[1:] - z[:-1]
        self.assertTrue(tf.reduce_all(diffs > 0).numpy())

    def test_higher_persistence_concentrates_diagonal(self):
        """Test that higher rho puts more weight on diagonal."""
        # Use same sigma for fair comparison
        sigma = 0.02
        _, P_low = tauchen_discretization(n=7, rho=0.5, sigma=sigma)
        _, P_high = tauchen_discretization(n=7, rho=0.95, sigma=sigma)
        
        diag_low = tf.linalg.diag_part(P_low)
        diag_high = tf.linalg.diag_part(P_high)
        
        # Higher persistence should have higher diagonal entries on average
        self.assertGreater(
            tf.reduce_mean(diag_high).numpy(),
            tf.reduce_mean(diag_low).numpy()
        )

    def test_zero_persistence_uniform_rows(self):
        """Test that rho=0 gives identical rows (iid process)."""
        # For rho=0, the process is iid so transition doesn't depend on current state
        # Use small rho instead of exactly 0 to avoid numerical issues with std_y calculation
        z, P = tauchen_discretization(n=7, rho=1e-10, sigma=0.1)
        
        # All rows should be nearly identical for iid process
        for i in range(1, 7):
            np.testing.assert_allclose(
                P[i].numpy(), P[0].numpy(), rtol=1e-4
            )

    def test_larger_sigma_spreads_grid(self):
        """Test that larger sigma creates a wider grid."""
        z_small, _ = tauchen_discretization(n=9, rho=0.9, sigma=0.01)
        z_large, _ = tauchen_discretization(n=9, rho=0.9, sigma=0.10)
        
        # Larger sigma should produce wider range in z
        range_small = z_small[-1] - z_small[0]
        range_large = z_large[-1] - z_large[0]
        
        self.assertGreater(range_large.numpy(), range_small.numpy())

    def test_stationary_distribution_exists(self):
        """Test that the transition matrix has a stationary distribution."""
        z, P = tauchen_discretization(n=7, rho=0.9, sigma=0.02)
        
        # Find stationary distribution by repeated multiplication
        pi = tf.ones(7, dtype=P.dtype) / 7.0
        for _ in range(1000):
            pi = tf.linalg.matvec(tf.transpose(P), pi)
        
        # Check it's actually stationary: pi @ P = pi
        pi_next = tf.linalg.matvec(tf.transpose(P), pi)
        np.testing.assert_allclose(pi.numpy(), pi_next.numpy(), rtol=1e-5)

    def test_transition_matrix_irreducible(self):
        """Test that all states can be reached (no zero rows/columns)."""
        z, P = tauchen_discretization(n=7, rho=0.9, sigma=0.05)
        
        # Each row should have positive mass somewhere
        row_maxes = tf.reduce_max(P, axis=1)
        self.assertTrue(tf.reduce_all(row_maxes > 0).numpy())
        
        # Each column should receive positive mass from somewhere
        col_sums = tf.reduce_sum(P, axis=0)
        self.assertTrue(tf.reduce_all(col_sums > 0).numpy())


class TestTauchenEdgeCases(unittest.TestCase):
    """Edge case tests for Tauchen discretization."""

    def test_small_grid(self):
        """Test with minimum viable grid size."""
        z, P = tauchen_discretization(n=3, rho=0.9, sigma=0.02)
        
        self.assertEqual(z.shape, (3,))
        self.assertEqual(P.shape, (3, 3))
        np.testing.assert_allclose(
            tf.reduce_sum(P, axis=1).numpy(),
            np.ones(3),
            rtol=1e-6
        )

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

    def test_high_persistence(self):
        """Test with very high persistence parameter."""
        z, P = tauchen_discretization(n=9, rho=0.99, sigma=0.02)
        
        # Should still produce valid probability matrix
        np.testing.assert_allclose(
            tf.reduce_sum(P, axis=1).numpy(),
            np.ones(9),
            rtol=1e-6
        )
        self.assertTrue(tf.reduce_all(P >= 0).numpy())

    def test_different_grid_sizes_consistency(self):
        """Test that finer grids approximate the same process."""
        # The middle point should always be exp(0) = 1 for odd grids
        for n in [5, 9, 15, 21]:
            z, _ = tauchen_discretization(n=n, rho=0.9, sigma=0.02)
            middle_idx = n // 2
            np.testing.assert_allclose(z[middle_idx].numpy(), 1.0, rtol=1e-6)


class TestTauchenNumericalProperties(unittest.TestCase):
    """Tests for numerical properties of the discretization."""

    def test_dtype_consistency(self):
        """Test that outputs have consistent dtype."""
        z, P = tauchen_discretization(n=7, rho=0.9, sigma=0.02)
        
        self.assertEqual(z.dtype, P.dtype)

    def test_no_nan_values(self):
        """Test that outputs contain no NaN values."""
        z, P = tauchen_discretization(n=11, rho=0.9, sigma=0.02)
        
        self.assertFalse(tf.reduce_any(tf.math.is_nan(z)).numpy())
        self.assertFalse(tf.reduce_any(tf.math.is_nan(P)).numpy())

    def test_no_inf_values(self):
        """Test that outputs contain no infinite values."""
        z, P = tauchen_discretization(n=11, rho=0.9, sigma=0.02)
        
        self.assertFalse(tf.reduce_any(tf.math.is_inf(z)).numpy())
        self.assertFalse(tf.reduce_any(tf.math.is_inf(P)).numpy())


if __name__ == '__main__':
    unittest.main()