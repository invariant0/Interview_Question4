# econ_models/vfi/engine.py
"""
Numerical engine for Value Function Iteration (VFI).

This module provides a generic fixed-point iterator for Bellman equations,
handling the maximization step and convergence checks. It is agnostic to
the specific economic model being solved.

Example:
    >>> engine = VFIEngine(beta=0.96, transition_matrix=P, tol=1e-7, max_iter=1000)
    >>> v_star = engine.run_vfi(v_init, flow_matrix)
"""

import tensorflow as tf

from econ_models.core.types import TENSORFLOW_DTYPE, Tensor


class VFIEngine:
    """
    Fixed-point iterator for Bellman equations.

    Iterates V_{t+1} = T(V_t) until ||V_{t+1} - V_t||_inf < tol.

    Attributes:
        beta: Discount factor.
        transition_matrix: Markov transition matrix for exogenous states.
        tol: Convergence tolerance (infinity norm).
        max_iter: Maximum iterations allowed.
    """

    def __init__(
        self,
        beta: float,
        transition_matrix: Tensor,
        tol: float,
        max_iter: int
    ) -> None:
        """
        Initialize the VFI engine.

        Args:
            beta: Discount factor.
            transition_matrix: Markov transition matrix for exogenous states.
            tol: Convergence tolerance (infinity norm).
            max_iter: Maximum number of iterations.
        """
        self.beta = tf.cast(beta, TENSORFLOW_DTYPE)
        self.transition_matrix = tf.cast(transition_matrix, TENSORFLOW_DTYPE)
        self.tol = float(tol)
        self.max_iter = int(max_iter)

    def run_vfi(self, v_init: Tensor, flow_matrix: Tensor) -> Tensor:
        """
        Execute value function iteration until convergence.

        Args:
            v_init: Initial guess for the value function.
            flow_matrix: Pre-computed flow utility matrix.

        Returns:
            Converged value function tensor.

        Raises:
            ValueError: If the rank of v_init is not supported (must be 2 or 3).
        """
        v_curr = tf.cast(v_init, TENSORFLOW_DTYPE)
        flow_matrix = tf.cast(flow_matrix, TENSORFLOW_DTYPE)
        rank = v_curr.shape.rank

        step_fn = self._select_step_function(rank)

        for _ in range(self.max_iter):
            v_next = step_fn(v_curr, flow_matrix)
            diff = tf.reduce_max(tf.abs(v_next - v_curr))
            v_curr = v_next

            if float(diff.numpy()) < self.tol:
                break

        return v_curr

    def _select_step_function(self, rank: int):
        """Select the appropriate Bellman step function based on dimensionality."""
        if rank == 2:
            return self._bellman_step_2d
        elif rank == 3:
            return self._bellman_step_3d
        else:
            raise ValueError(
                f"Unsupported value function rank: {rank}. Expected 2 or 3."
            )

    @tf.function
    def _bellman_step_2d(self, v_old: Tensor, flow: Tensor) -> Tensor:
        """
        Perform single Bellman update for 2D state (K, Z).

        Args:
            v_old: Value function [n_capital, n_productivity].
            flow: Payoff matrix [n_capital, n_capital, n_productivity].

        Returns:
            Updated value function [n_capital, n_productivity].
        """
        expected_value = tf.matmul(v_old, self.transition_matrix, transpose_b=True)
        rhs = flow + self.beta * expected_value[None, :, :]
        v_new = tf.reduce_max(rhs, axis=1)
        return tf.maximum(v_new, tf.cast(0.0, TENSORFLOW_DTYPE))

    @tf.function
    def _bellman_step_3d(self, v_old: Tensor, flow: Tensor) -> Tensor:
        """
        Perform single Bellman update for 3D state (K, B, Z).

        Args:
            v_old: Value function [n_capital, n_debt, n_productivity].
            flow: Payoff matrix [n_capital, n_debt, n_capital, n_debt, n_productivity].

        Returns:
            Updated value function [n_capital, n_debt, n_productivity].
        """
        expected_value = tf.tensordot(
            v_old, self.transition_matrix, axes=[[2], [1]]
        )
        rhs = flow + self.beta * expected_value[None, None, :, :, :]
        v_new = tf.reduce_max(rhs, axis=[2, 3])
        return tf.maximum(v_new, tf.cast(0.0, TENSORFLOW_DTYPE))