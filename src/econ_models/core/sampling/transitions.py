# econ_models/core/sampling/transitions.py
"""
Stochastic transition functions for productivity processes.

This module implements the log-AR(1) transition dynamics used
for productivity evolution in the economic models.
"""

import tensorflow as tf

from econ_models.core.types import TENSORFLOW_DTYPE, Tensor


class TransitionFunctions:
    """Static methods for stochastic state-variable transitions.

    Implement the log-AR(1) transition dynamics used for productivity
    evolution in the economic models.
    """

    @staticmethod
    def log_ar1_transition(
        z_curr: Tensor,
        rho: float,
        epsilon: Tensor
    ) -> Tensor:
        """
        Compute next state for a log-AR(1) process.

        Formula:
            ln(z') = rho * ln(z) + epsilon
            z' = exp(ln(z'))

        Args:
            z_curr: Current productivity level.
            rho: Persistence parameter.
            epsilon: Random shock (innovation).

        Returns:
            Next period productivity level.
        """
        rho_t = tf.cast(rho, TENSORFLOW_DTYPE)
        epsilon = tf.cast(epsilon, TENSORFLOW_DTYPE)

        # Ensure z is positive before taking log
        z_safe = tf.maximum(z_curr, tf.cast(1e-12, TENSORFLOW_DTYPE))

        ln_z = tf.math.log(z_safe)
        ln_z_prime = rho_t * ln_z + epsilon

        return tf.exp(ln_z_prime)