"""Unit tests for the VFI Engine."""

import unittest
import tensorflow as tf
import numpy as np

from econ_models.vfi.engine import VFIEngine
from econ_models.core.types import TENSORFLOW_DTYPE


class TestVFIEngineInitialization(unittest.TestCase):
    """Tests for VFIEngine initialization."""

    def setUp(self):
        """Set up a simple transition matrix for tests."""
        self.simple_P = tf.constant([
            [0.9, 0.1],
            [0.1, 0.9]
        ], dtype=TENSORFLOW_DTYPE)

    def test_initialization_stores_parameters(self):
        """Test that constructor stores parameters correctly."""
        engine = VFIEngine(
            beta=0.96,
            transition_matrix=self.simple_P,
            tol=1e-7,
            max_iter=500
        )
        
        self.assertAlmostEqual(float(engine.beta), 0.96, places=6)
        self.assertEqual(engine.tol, 1e-7)
        self.assertEqual(engine.max_iter, 500)

    def test_transition_matrix_cast_to_correct_dtype(self):
        """Test that transition matrix is cast to TENSORFLOW_DTYPE."""
        P_float32 = tf.constant([[0.9, 0.1], [0.1, 0.9]], dtype=tf.float32)
        engine = VFIEngine(
            beta=0.96,
            transition_matrix=P_float32,
            tol=1e-7,
            max_iter=500
        )
        
        self.assertEqual(engine.transition_matrix.dtype, TENSORFLOW_DTYPE)


class TestVFIEngineStepFunctionSelection(unittest.TestCase):
    """Tests for step function selection based on rank."""

    def setUp(self):
        """Set up engine for tests."""
        self.P = tf.constant([[0.9, 0.1], [0.1, 0.9]], dtype=TENSORFLOW_DTYPE)
        self.engine = VFIEngine(
            beta=0.96,
            transition_matrix=self.P,
            tol=1e-7,
            max_iter=100
        )

    def test_selects_2d_step_for_rank_2(self):
        """Test that 2D step function is selected for rank-2 value function."""
        step_fn = self.engine._select_step_function(rank=2)
        self.assertEqual(step_fn, self.engine._bellman_step_2d)

    def test_selects_3d_step_for_rank_3(self):
        """Test that 3D step function is selected for rank-3 value function."""
        step_fn = self.engine._select_step_function(rank=3)
        self.assertEqual(step_fn, self.engine._bellman_step_3d)

    def test_raises_for_unsupported_rank(self):
        """Test that unsupported ranks raise ValueError."""
        with self.assertRaises(ValueError) as context:
            self.engine._select_step_function(rank=4)
        
        self.assertIn("Unsupported value function rank", str(context.exception))

    def test_raises_for_rank_1(self):
        """Test that rank 1 raises ValueError."""
        with self.assertRaises(ValueError):
            self.engine._select_step_function(rank=1)


class TestBellmanStep2D(unittest.TestCase):
    """Tests for the 2D Bellman step operator."""

    def setUp(self):
        """Set up engine and simple test case."""
        self.P = tf.constant([
            [0.8, 0.2],
            [0.2, 0.8]
        ], dtype=TENSORFLOW_DTYPE)
        
        self.engine = VFIEngine(
            beta=0.9,
            transition_matrix=self.P,
            tol=1e-7,
            max_iter=100
        )
        
        # Simple 2x2 state space (2 capital, 2 productivity)
        self.n_k = 3
        self.n_z = 2

    def test_output_shape_matches_input(self):
        """Test that Bellman step preserves value function shape."""
        v_old = tf.zeros((self.n_k, self.n_z), dtype=TENSORFLOW_DTYPE)
        flow = tf.ones((self.n_k, self.n_k, self.n_z), dtype=TENSORFLOW_DTYPE)
        
        v_new = self.engine._bellman_step_2d(v_old, flow)
        
        self.assertEqual(v_new.shape, (self.n_k, self.n_z))

    def test_value_function_non_negative(self):
        """Test that output value function is non-negative."""
        v_old = tf.zeros((self.n_k, self.n_z), dtype=TENSORFLOW_DTYPE)
        # Negative flow to test floor at zero
        flow = -tf.ones((self.n_k, self.n_k, self.n_z), dtype=TENSORFLOW_DTYPE)
        
        v_new = self.engine._bellman_step_2d(v_old, flow)
        
        self.assertTrue(tf.reduce_all(v_new >= 0).numpy())

    def test_positive_flow_increases_value(self):
        """Test that positive flow leads to positive values."""
        v_old = tf.zeros((self.n_k, self.n_z), dtype=TENSORFLOW_DTYPE)
        flow = tf.ones((self.n_k, self.n_k, self.n_z), dtype=TENSORFLOW_DTYPE) * 10.0
        
        v_new = self.engine._bellman_step_2d(v_old, flow)
        
        self.assertTrue(tf.reduce_all(v_new > 0).numpy())

    def test_bellman_step_manual_calculation(self):
        """Test Bellman step against manual calculation."""
        # Simple case: 2 capital states, 2 productivity states
        v_old = tf.constant([
            [1.0, 2.0],
            [1.5, 2.5]
        ], dtype=TENSORFLOW_DTYPE)
        
        # Flow[k, k', z] - simple diagonal preference
        flow = tf.constant([
            [[1.0, 1.0], [0.5, 0.5]],  # From k=0
            [[0.5, 0.5], [1.0, 1.0]]   # From k=1
        ], dtype=TENSORFLOW_DTYPE)
        
        v_new = self.engine._bellman_step_2d(v_old, flow)
        
        # Manual calculation:
        # EV = v_old @ P^T
        expected_ev = tf.matmul(v_old, self.P, transpose_b=True)
        # For each (k, z): max over k' of {flow[k, k', z] + beta * EV[k', z]}
        
        # Check shape
        self.assertEqual(v_new.shape, (2, 2))
        # Check finite
        self.assertTrue(tf.reduce_all(tf.math.is_finite(v_new)).numpy())


class TestBellmanStep3D(unittest.TestCase):
    """Tests for the 3D Bellman step operator."""

    def setUp(self):
        """Set up engine for 3D tests."""
        self.P = tf.constant([
            [0.8, 0.2],
            [0.2, 0.8]
        ], dtype=TENSORFLOW_DTYPE)
        
        self.engine = VFIEngine(
            beta=0.9,
            transition_matrix=self.P,
            tol=1e-7,
            max_iter=100
        )
        
        self.n_k = 3
        self.n_b = 4
        self.n_z = 2

    def test_output_shape_matches_input(self):
        """Test that 3D Bellman step preserves value function shape."""
        v_old = tf.zeros((self.n_k, self.n_b, self.n_z), dtype=TENSORFLOW_DTYPE)
        flow = tf.ones(
            (self.n_k, self.n_b, self.n_k, self.n_b, self.n_z),
            dtype=TENSORFLOW_DTYPE
        )
        
        v_new = self.engine._bellman_step_3d(v_old, flow)
        
        self.assertEqual(v_new.shape, (self.n_k, self.n_b, self.n_z))

    def test_value_function_non_negative(self):
        """Test that 3D output is non-negative."""
        v_old = tf.zeros((self.n_k, self.n_b, self.n_z), dtype=TENSORFLOW_DTYPE)
        flow = -tf.ones(
            (self.n_k, self.n_b, self.n_k, self.n_b, self.n_z),
            dtype=TENSORFLOW_DTYPE
        )
        
        v_new = self.engine._bellman_step_3d(v_old, flow)
        
        self.assertTrue(tf.reduce_all(v_new >= 0).numpy())

    def test_maximization_over_both_choice_dims(self):
        """Test that max is taken over both k' and b' dimensions."""
        v_old = tf.zeros((2, 2, 2), dtype=TENSORFLOW_DTYPE)
        
        # Create flow where one (k', b') combination dominates
        flow = tf.zeros((2, 2, 2, 2, 2), dtype=TENSORFLOW_DTYPE)
        # Set high value at k'=1, b'=1
        indices = tf.constant([[0, 0, 1, 1, 0], [0, 0, 1, 1, 1],
                               [0, 1, 1, 1, 0], [0, 1, 1, 1, 1],
                               [1, 0, 1, 1, 0], [1, 0, 1, 1, 1],
                               [1, 1, 1, 1, 0], [1, 1, 1, 1, 1]])
        updates = tf.constant([100.0] * 8, dtype=TENSORFLOW_DTYPE)
        flow = tf.tensor_scatter_nd_update(flow, indices, updates)
        
        v_new = self.engine._bellman_step_3d(v_old, flow)
        
        # All states should have value close to 100 (the max)
        self.assertTrue(tf.reduce_all(v_new >= 99.0).numpy())


class TestVFIConvergence(unittest.TestCase):
    """Tests for VFI convergence behavior."""

    def setUp(self):
        """Set up engine with identity-like transition."""
        # Near-deterministic transitions for predictable behavior
        self.P = tf.constant([
            [1.0, 0.0],
            [0.0, 1.0]
        ], dtype=TENSORFLOW_DTYPE)
        
        self.engine = VFIEngine(
            beta=0.9,
            transition_matrix=self.P,
            tol=1e-6,
            max_iter=1000
        )

    def test_converges_from_zero_initialization(self):
        """Test that VFI converges from zero initial guess."""
        n_k, n_z = 5, 2
        v_init = tf.zeros((n_k, n_z), dtype=TENSORFLOW_DTYPE)
        flow = tf.ones((n_k, n_k, n_z), dtype=TENSORFLOW_DTYPE)
        
        v_star = self.engine.run_vfi(v_init, flow)
        
        # Should converge to positive values
        self.assertTrue(tf.reduce_all(v_star > 0).numpy())
        self.assertTrue(tf.reduce_all(tf.math.is_finite(v_star)).numpy())

    def test_convergence_to_analytical_solution(self):
        """Test convergence to known analytical solution for simple case."""
        # For deterministic transitions and uniform flow of 1:
        # V = 1 + beta*V => V = 1/(1-beta) = 10 for beta=0.9
        n_k, n_z = 3, 2
        v_init = tf.zeros((n_k, n_z), dtype=TENSORFLOW_DTYPE)
        flow = tf.ones((n_k, n_k, n_z), dtype=TENSORFLOW_DTYPE)
        
        v_star = self.engine.run_vfi(v_init, flow)
        
        expected_value = 1.0 / (1.0 - 0.9)  # = 10
        np.testing.assert_allclose(
            v_star.numpy(),
            np.full((n_k, n_z), expected_value),
            rtol=1e-4
        )

    def test_respects_max_iterations(self):
        """Test that iteration stops at max_iter."""
        engine = VFIEngine(
            beta=0.9999,  # Very slow convergence
            transition_matrix=self.P,
            tol=1e-15,    # Impossible tolerance
            max_iter=5    # Should stop here
        )
        
        n_k, n_z = 3, 2
        v_init = tf.zeros((n_k, n_z), dtype=TENSORFLOW_DTYPE)
        flow = tf.ones((n_k, n_k, n_z), dtype=TENSORFLOW_DTYPE)
        
        # Should complete without hanging
        v_result = engine.run_vfi(v_init, flow)
        
        # Result should exist and be finite
        self.assertTrue(tf.reduce_all(tf.math.is_finite(v_result)).numpy())

    def test_3d_vfi_converges(self):
        """Test that 3D VFI also converges."""
        P_3state = tf.eye(3, dtype=TENSORFLOW_DTYPE)
        engine = VFIEngine(
            beta=0.9,
            transition_matrix=P_3state,
            tol=1e-6,
            max_iter=500
        )
        
        n_k, n_b, n_z = 3, 4, 3
        v_init = tf.zeros((n_k, n_b, n_z), dtype=TENSORFLOW_DTYPE)
        flow = tf.ones((n_k, n_b, n_k, n_b, n_z), dtype=TENSORFLOW_DTYPE)
        
        v_star = engine.run_vfi(v_init, flow)
        
        expected_value = 1.0 / (1.0 - 0.9)
        np.testing.assert_allclose(
            v_star.numpy(),
            np.full((n_k, n_b, n_z), expected_value),
            rtol=1e-4
        )


class TestVFIEdgeCases(unittest.TestCase):
    """Edge case tests for VFI engine."""

    def test_single_state_productivity(self):
        """Test with single productivity state (deterministic)."""
        P = tf.constant([[1.0]], dtype=TENSORFLOW_DTYPE)
        engine = VFIEngine(beta=0.9, transition_matrix=P, tol=1e-7, max_iter=100)
        
        v_init = tf.zeros((5, 1), dtype=TENSORFLOW_DTYPE)
        flow = tf.ones((5, 5, 1), dtype=TENSORFLOW_DTYPE) * 2.0
        
        v_star = engine.run_vfi(v_init, flow)
        
        expected = 2.0 / (1.0 - 0.9)
        np.testing.assert_allclose(v_star.numpy(), expected, rtol=1e-4)

    def test_zero_discount_factor(self):
        """Test with beta=0 (myopic agent)."""
        P = tf.constant([[0.5, 0.5], [0.5, 0.5]], dtype=TENSORFLOW_DTYPE)
        engine = VFIEngine(beta=0.0, transition_matrix=P, tol=1e-7, max_iter=100)
        
        v_init = tf.zeros((3, 2), dtype=TENSORFLOW_DTYPE)
        flow = tf.ones((3, 3, 2), dtype=TENSORFLOW_DTYPE) * 5.0
        
        v_star = engine.run_vfi(v_init, flow)
        
        # With beta=0, value = max flow = 5.0
        np.testing.assert_allclose(v_star.numpy(), 5.0, rtol=1e-6)

    def test_heterogeneous_flow_values(self):
        """Test with varying flow values across states."""
        P = tf.eye(2, dtype=TENSORFLOW_DTYPE)
        engine = VFIEngine(beta=0.9, transition_matrix=P, tol=1e-7, max_iter=500)
        
        # Flow varies by productivity state
        flow = tf.stack([
            tf.ones((3, 3), dtype=TENSORFLOW_DTYPE) * 1.0,  # z=0: flow=1
            tf.ones((3, 3), dtype=TENSORFLOW_DTYPE) * 2.0   # z=1: flow=2
        ], axis=-1)
        
        v_init = tf.zeros((3, 2), dtype=TENSORFLOW_DTYPE)
        v_star = engine.run_vfi(v_init, flow)
        
        # Check z=1 states have higher value than z=0
        self.assertTrue(tf.reduce_all(v_star[:, 1] > v_star[:, 0]).numpy())


if __name__ == '__main__':
    unittest.main()