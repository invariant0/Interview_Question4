# econ_models/dl/training/loss_calculator.py
"""
Loss calculation utilities for economic deep learning.

This module provides loss functions for training neural networks
to satisfy economic equilibrium conditions.
"""

import tensorflow as tf

from econ_models.core.types import TENSORFLOW_DTYPE


class LossCalculator:
    """
    Static methods for computing training losses.

    These loss functions implement the All-in-One (AiO) loss formulation
    that uses two independent Monte Carlo samples to obtain unbiased
    estimates of squared residuals.
    """

    @staticmethod
    def bellman_aio_loss(
        value_curr: tf.Tensor,
        target_1: tf.Tensor,
        target_2: tf.Tensor
    ) -> tf.Tensor:
        """
        Compute All-in-One Bellman residual loss.

        This formulation uses the product of two independent residual
        estimates to obtain an unbiased estimate of E[R]^2.

        Args:
            value_curr: Current value function estimate V(s).
            target_1: First bootstrap target: r + beta * V(s')_1.
            target_2: Second bootstrap target: r + beta * V(s')_2.

        Returns:
            Scalar loss value.
        """
        bellman_res1 = value_curr - target_1
        bellman_res2 = value_curr - target_2
        return tf.reduce_mean(bellman_res1 * bellman_res2)

    @staticmethod
    def euler_aio_loss(
        foc_residual_1: tf.Tensor,
        foc_residual_2: tf.Tensor
    ) -> tf.Tensor:
        """
        Compute All-in-One first-order condition loss.

        Args:
            foc_residual_1: First FOC residual estimate.
            foc_residual_2: Second FOC residual estimate.

        Returns:
            Scalar loss value.
        """
        return tf.reduce_mean(foc_residual_1 * foc_residual_2)