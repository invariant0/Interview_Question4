# econ_models/core/grids/tauchen.py
"""
Tauchen method for AR(1) process discretization.

This module implements Tauchen's (1986) method for approximating
continuous AR(1) processes with discrete Markov chains.
"""

from typing import Tuple

import tensorflow as tf
import tensorflow_probability as tfp

from econ_models.core.types import TENSORFLOW_DTYPE, Tensor

tfd = tfp.distributions


def tauchen_discretization(
    n: int,
    rho: float,
    sigma: float,
    m: float = 3.0
) -> Tuple[Tensor, Tensor]:
    """
    Discretize an AR(1) process using Tauchen's method.

    The AR(1) process: ln(z') = rho * ln(z) + epsilon
    where epsilon ~ N(0, sigma^2)

    Args:
        n: Number of grid points.
        rho: Persistence parameter of the AR(1) process.
        sigma: Standard deviation of the innovation term.
        m: Width of the grid in standard deviations.

    Returns:
        Tuple containing:
            - z: Discretized state grid (in levels, not logs).
            - p_matrix: Transition probability matrix P[i,j] = Pr(z'=j|z=i).
    """
    n_int = int(n)
    rho_t = tf.cast(rho, TENSORFLOW_DTYPE)
    sigma_t = tf.cast(sigma, TENSORFLOW_DTYPE)
    m_t = tf.cast(m, TENSORFLOW_DTYPE)
    one = tf.cast(1.0, TENSORFLOW_DTYPE)

    # Unconditional standard deviation
    std_y = sigma_t / tf.sqrt(one - rho_t ** 2)
    x_max = m_t * std_y

    # Create equally-spaced grid in log space
    x = tf.linspace(-x_max, x_max, n_int)
    x = tf.cast(x, TENSORFLOW_DTYPE)
    step = x[1] - x[0]

    # Standard normal distribution for CDF calculations
    dist = tfd.Normal(loc=tf.cast(0.0, TENSORFLOW_DTYPE), scale=one)

    # Build transition matrix using broadcasting
    p_matrix = _build_transition_matrix(x, rho_t, sigma_t, step, dist)

    # Convert from log to level
    z = tf.exp(x)

    return tf.cast(z, TENSORFLOW_DTYPE), tf.cast(p_matrix, TENSORFLOW_DTYPE)


def _build_transition_matrix(
    x: Tensor,
    rho: Tensor,
    sigma: Tensor,
    step: Tensor,
    dist: tfd.Normal
) -> Tensor:
    """
    Build the Markov transition matrix for the discretized process.

    Args:
        x: Log-space grid points.
        rho: Persistence parameter.
        sigma: Innovation standard deviation.
        step: Grid spacing.
        dist: Standard normal distribution for CDF.

    Returns:
        Row-normalized transition probability matrix.
    """
    one = tf.cast(1.0, TENSORFLOW_DTYPE)

    # Broadcasting: x_j is next state, x_i is current state
    x_j = x[None, :]  # Shape: (1, n)
    x_i = x[:, None]  # Shape: (n, 1)

    # Standardized upper and lower bounds
    upper = (x_j + step / 2.0 - rho * x_i) / sigma
    lower = (x_j - step / 2.0 - rho * x_i) / sigma

    # Middle columns: probability between bounds
    p_middle = dist.cdf(upper) - dist.cdf(lower)

    # First column: cumulative up to first state + half step
    p_col0 = dist.cdf((x[0] + step / 2.0 - rho * x_i) / sigma)

    # Last column: 1 - CDF up to last state - half step
    p_coln = one - dist.cdf((x[-1] - step / 2.0 - rho * x_i) / sigma)

    # Assemble matrix
    p_matrix = tf.concat([p_col0, p_middle[:, 1:-1], p_coln], axis=1)

    # Normalize rows to ensure valid probability distribution
    row_sums = tf.reduce_sum(p_matrix, axis=1, keepdims=True)
    p_matrix = p_matrix / row_sums

    return p_matrix


# Backward compatibility alias
tauchen = tauchen_discretization