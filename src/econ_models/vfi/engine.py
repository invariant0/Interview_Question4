"""Numerical engine for Value Function Iteration (VFI).

This module provides a generic fixed-point iterator for Bellman equations,
handling the maximisation step and convergence checks.  It is agnostic to
the specific economic model being solved.

Example::

    >>> engine = VFIEngine(beta=0.96, transition_matrix=P, tol=1e-7, max_iter=1000)
    >>> v_star = engine.run_vfi(v_init, flow_matrix)
"""

from __future__ import annotations

import logging
from typing import Callable

import tensorflow as tf

from econ_models.core.types import TENSORFLOW_DTYPE, Tensor

logger = logging.getLogger(__name__)


class VFIEngine:
    """Fixed-point iterator for Bellman equations.

    Iterates :math:`V_{t+1} = T(V_t)` until
    :math:`\\|V_{t+1} - V_t\\|_\\infty < \\text{tol}`.

    Parameters
    ----------
    beta : float
        Discount factor in (0, 1).
    transition_matrix : Tensor
        Markov transition matrix for exogenous states, shape ``(n_z, n_z)``.
    tol : float
        Convergence tolerance (sup-norm).
    max_iter : int
        Maximum number of Bellman iterations.

    Raises
    ------
    ValueError
        If *beta* is not in the open interval (0, 1).
    ValueError
        If *tol* is non-positive.
    ValueError
        If *max_iter* is non-positive.
    """

    def __init__(
        self,
        beta: float,
        transition_matrix: Tensor,
        tol: float,
        max_iter: int,
    ) -> None:
        if not 0.0 < beta < 1.0:
            raise ValueError(
                f"Discount factor must be in (0, 1), got {beta}."
            )
        if tol <= 0.0:
            raise ValueError(f"Tolerance must be positive, got {tol}.")
        if max_iter <= 0:
            raise ValueError(
                f"max_iter must be positive, got {max_iter}."
            )

        self.beta: tf.Tensor = tf.cast(beta, TENSORFLOW_DTYPE)
        self.transition_matrix: tf.Tensor = tf.cast(
            transition_matrix, TENSORFLOW_DTYPE
        )
        self.tol: float = float(tol)
        self.max_iter: int = int(max_iter)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_vfi(self, v_init: Tensor, flow_matrix: Tensor) -> Tensor:
        """Execute value function iteration until convergence.

        Parameters
        ----------
        v_init : Tensor
            Initial guess for the value function (rank 2 or 3).
        flow_matrix : Tensor
            Pre-computed flow utility matrix.

        Returns
        -------
        Tensor
            Converged value function with the same shape as *v_init*.

        Raises
        ------
        ValueError
            If *v_init* has an unsupported rank (must be 2 or 3).
        """
        v_curr = tf.cast(v_init, TENSORFLOW_DTYPE)
        flow_matrix = tf.cast(flow_matrix, TENSORFLOW_DTYPE)
        rank = v_curr.shape.rank

        step_fn = self._select_step_function(rank)

        for iteration in range(self.max_iter):
            v_next = step_fn(v_curr, flow_matrix)
            diff = tf.reduce_max(tf.abs(v_next - v_curr))
            v_curr = v_next

            if float(diff.numpy()) < self.tol:
                logger.info(
                    "VFIEngine converged in %d iterations (diff=%.2e).",
                    iteration + 1,
                    float(diff.numpy()),
                )
                break
        else:
            logger.warning(
                "VFIEngine did not converge after %d iterations "
                "(final diff=%.2e).",
                self.max_iter,
                float(diff.numpy()),
            )

        return v_curr

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _select_step_function(
        self, rank: int
    ) -> Callable[[Tensor, Tensor], Tensor]:
        """Return the Bellman step function matching the value-function rank.

        Parameters
        ----------
        rank : int
            Rank of the value-function tensor (2 or 3).

        Returns
        -------
        Callable
            The appropriate Bellman-step method.

        Raises
        ------
        ValueError
            If *rank* is not 2 or 3.
        """
        if rank == 2:
            return self._bellman_step_2d
        if rank == 3:
            return self._bellman_step_3d
        raise ValueError(
            f"Unsupported value function rank: {rank}. Expected 2 or 3."
        )

    @tf.function
    def _bellman_step_2d(self, v_old: Tensor, flow: Tensor) -> Tensor:
        """Single Bellman update for 2-D state space (K, Z).

        Parameters
        ----------
        v_old : Tensor
            Value function, shape ``[n_capital, n_productivity]``.
        flow : Tensor
            Flow utility, shape ``[n_capital, n_capital, n_productivity]``.

        Returns
        -------
        Tensor
            Updated value function, shape ``[n_capital, n_productivity]``.
        """
        expected_value = tf.matmul(
            v_old, self.transition_matrix, transpose_b=True
        )
        rhs = flow + self.beta * expected_value[None, :, :]
        v_new = tf.reduce_max(rhs, axis=1)
        return tf.maximum(v_new, tf.cast(0.0, TENSORFLOW_DTYPE))

    @tf.function
    def _bellman_step_3d(self, v_old: Tensor, flow: Tensor) -> Tensor:
        """Single Bellman update for 3-D state space (K, B, Z).

        Parameters
        ----------
        v_old : Tensor
            Value function, shape ``[n_capital, n_debt, n_productivity]``.
        flow : Tensor
            Flow utility, shape
            ``[n_capital, n_debt, n_capital, n_debt, n_productivity]``.

        Returns
        -------
        Tensor
            Updated value function, shape
            ``[n_capital, n_debt, n_productivity]``.
        """
        expected_value = tf.tensordot(
            v_old, self.transition_matrix, axes=[[2], [1]]
        )
        rhs = flow + self.beta * expected_value[None, None, :, :, :]
        v_new = tf.reduce_max(rhs, axis=[2, 3])
        return tf.maximum(v_new, tf.cast(0.0, TENSORFLOW_DTYPE))