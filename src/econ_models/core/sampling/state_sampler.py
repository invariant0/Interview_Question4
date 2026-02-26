# econ_models/core/sampling/state_sampler.py
"""
State variable sampling for deep learning training.

This module provides functions for generating random samples of
state variables with optional curriculum learning support.
"""

from typing import Tuple, Optional

import tensorflow as tf
import tensorflow_probability as tfp

from econ_models.core.types import TENSORFLOW_DTYPE

tfd = tfp.distributions


class StateSampler:
    """Sample economic state variables for neural network training.

    Provide CPU (Beta-distribution-based) and GPU (XLA-compatible
    uniform) sampling strategies for capital, productivity, and
    optionally debt state variables.
    """

    @staticmethod
    def sample_states(
        batch_size: int,
        bonds_config: dict,
        include_debt: bool = False,
    ) -> Tuple[tf.Tensor, ...]:
        """Generate random samples of state variables (K, Z, [B]).

        Use scaled Beta(1, 1) distributions (equivalent to uniform)
        for each variable, scaled to the domain bounds in *bonds_config*.

        Args:
            batch_size: Number of samples to generate.
            bonds_config: Dictionary with sampling bounds (k_min,
                k_max, z_min, z_max, and optionally b_min, b_max).
            include_debt: Whether to include debt as a state variable.

        Returns:
            Tuple of tensors each with shape ``(batch_size, 1)``:
                - Without debt: (K, Z)
                - With debt: (K, B, Z)
        """

        k_samples = StateSampler._sample_capital(
            batch_size, bonds_config
        )
        z_samples = StateSampler._sample_productivity(
            batch_size, bonds_config
        )

        if not include_debt:
            return k_samples, z_samples

        b_samples = StateSampler._sample_debt(
            batch_size, bonds_config
        )
        return k_samples, b_samples, z_samples

    @staticmethod
    def sample_states_gpu(
        batch_size: int,
        bonds_config: dict,
        include_debt: bool = False,
    ) -> Tuple[tf.Tensor, ...]:
        """
        Generate random state samples using tf.random.uniform (XLA-compatible).

        This is the GPU-friendly counterpart of sample_states(). Uses
        tf.random.uniform instead of tfp.distributions.Beta for full XLA
        compatibility and direct GPU execution without CPU→GPU transfer.

        With the default Beta(1,1) parameters in sample_states, the two
        methods produce identically distributed samples.

        Args:
            batch_size: Number of samples to generate.
            bonds_config: Dict with keys k_min, k_max, z_min, z_max,
                and optionally b_min, b_max.
            include_debt: Whether to include debt as a state variable.

        Returns:
            Tuple of tensors with shape (batch_size, 1):
                - Without debt: (K, Z)
                - With debt: (K, B, Z)
        """
        shape = (batch_size, 1)

        k = tf.random.uniform(
            shape,
            minval=bonds_config['k_min'],
            maxval=bonds_config['k_max'],
            dtype=TENSORFLOW_DTYPE,
        )
        z = tf.random.uniform(
            shape,
            minval=bonds_config['z_min'],
            maxval=bonds_config['z_max'],
            dtype=TENSORFLOW_DTYPE,
        )

        if not include_debt:
            return k, z

        b = tf.random.uniform(
            shape,
            minval=bonds_config['b_min'],
            maxval=bonds_config['b_max'],
            dtype=TENSORFLOW_DTYPE,
        )
        return k, b, z

    @staticmethod
    def _sample_capital(
        batch_size: tf.Tensor,
        bonds_config: dict,
    ) -> tf.Tensor:
        """Sample capital values with curriculum bounds."""
        k_min, k_max = bonds_config['k_min'], bonds_config['k_max']
        samples = StateSampler._sample_beta_scaled(
            shape=(batch_size,), minval=k_min, maxval=k_max
        )
        return tf.reshape(samples, (-1, 1))

    @staticmethod
    def _sample_productivity(
        batch_size: tf.Tensor,
        bonds_config: dict,
    ) -> tf.Tensor:
        """Sample productivity values with curriculum bounds."""
        z_min, z_max = bonds_config['z_min'], bonds_config['z_max']
        samples = StateSampler._sample_beta_scaled(
            shape=(batch_size,), minval=z_min, maxval=z_max
        )
        return tf.reshape(samples, (-1, 1))

    @staticmethod
    def _sample_debt(
        batch_size: tf.Tensor,
        bonds_config: dict,
    ) -> tf.Tensor:
        """Sample debt values with curriculum bounds."""
        b_min, b_max = bonds_config['b_min'], bonds_config['b_max']
        samples = StateSampler._sample_beta_scaled(
            shape=(batch_size,), minval=b_min, maxval=b_max
        )
        return tf.reshape(samples, (-1, 1))

    @staticmethod
    def _sample_beta_scaled(
        shape: Tuple[int, ...],
        minval: tf.Tensor,
        maxval: tf.Tensor,
        alpha: float = 1.0,
        beta: float = 1.0
    ) -> tf.Tensor:
        """
        Sample from a scaled Beta distribution.

        With alpha=beta=1.0, this is equivalent to uniform sampling.

        Args:
            shape: Output shape.
            minval: Minimum value.
            maxval: Maximum value.
            alpha: Beta distribution alpha parameter.
            beta: Beta distribution beta parameter.

        Returns:
            Samples scaled to [minval, maxval].
        """
        alpha_t = tf.cast(alpha, TENSORFLOW_DTYPE)
        beta_t = tf.cast(beta, TENSORFLOW_DTYPE)
        dist = tfd.Beta(concentration1=alpha_t, concentration0=beta_t)
        beta_samples = dist.sample(shape)
        return minval + (maxval - minval) * beta_samples