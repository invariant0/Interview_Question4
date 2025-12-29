# econ_models/core/sampling/state_sampler.py
"""
State variable sampling for deep learning training.

This module provides functions for generating random samples of
state variables with optional curriculum learning support.
"""

from typing import Tuple, Optional

import tensorflow as tf
import tensorflow_probability as tfp

from econ_models.config.dl_config import DeepLearningConfig
from econ_models.core.types import TENSORFLOW_DTYPE
from econ_models.core.sampling.curriculum import CurriculumBounds

tfd = tfp.distributions


class StateSampler:
    """Utilities for sampling economic state variables."""

    @staticmethod
    def sample_states(
        batch_size: int,
        config: DeepLearningConfig,
        include_debt: bool = False,
        progress: Optional[tf.Tensor] = None
    ) -> Tuple[tf.Tensor, ...]:
        """
        Generate random samples of state variables (K, Z, [B]).

        Sample size increases with progress (curriculum learning).
        Productivity is sampled to emphasize the full support.

        Args:
            batch_size: Maximum samples at full progress.
            config: Configuration with domain bounds.
            include_debt: Whether to include debt as a state variable.
            progress: Curriculum progress in [0, 1].

        Returns:
            Tuple of tensors with shape (current_batch_size, 1):
                - Without debt: (K, Z)
                - With debt: (K, B, Z)
        """
        progress = StateSampler._normalize_progress(progress)
        current_batch_size = StateSampler._calculate_batch_size(
            batch_size, progress
        )

        k_samples = StateSampler._sample_capital(
            current_batch_size, config, progress
        )
        z_samples = StateSampler._sample_productivity(
            current_batch_size, config, progress
        )

        if not include_debt:
            return k_samples, z_samples

        b_samples = StateSampler._sample_debt(
            current_batch_size, config, progress
        )
        return k_samples, b_samples, z_samples

    @staticmethod
    def _normalize_progress(progress: Optional[tf.Tensor]) -> tf.Tensor:
        """Ensure progress is a valid tensor."""
        if progress is None:
            return tf.cast(1.0, TENSORFLOW_DTYPE)
        return tf.cast(progress, TENSORFLOW_DTYPE)

    @staticmethod
    def _calculate_batch_size(
        max_batch_size: int,
        progress: tf.Tensor
    ) -> tf.Tensor:
        """Calculate current batch size based on progress."""
        min_batch_ratio = 0.1
        batch_ratio = min_batch_ratio + (1.0 - min_batch_ratio) * progress
        current_size = tf.cast(
            tf.cast(max_batch_size, TENSORFLOW_DTYPE) * batch_ratio,
            tf.int32
        )
        return tf.maximum(current_size, 16)

    @staticmethod
    def _sample_capital(
        batch_size: tf.Tensor,
        config: DeepLearningConfig,
        progress: tf.Tensor
    ) -> tf.Tensor:
        """Sample capital values with curriculum bounds."""
        k_min, k_max = CurriculumBounds.get_curriculum_bounds(
            tf.cast(config.capital_min, TENSORFLOW_DTYPE),
            tf.cast(config.capital_max, TENSORFLOW_DTYPE),
            tf.cast(config.capital_steady_state, TENSORFLOW_DTYPE),
            progress,
            config.curriculum_initial_ratio
        )
        samples = StateSampler._sample_beta_scaled(
            shape=(batch_size,), minval=k_min, maxval=k_max
        )
        return tf.reshape(samples, (-1, 1))

    @staticmethod
    def _sample_productivity(
        batch_size: tf.Tensor,
        config: DeepLearningConfig,
        progress: tf.Tensor
    ) -> tf.Tensor:
        """Sample productivity values with curriculum bounds."""
        z_min, z_max = CurriculumBounds.get_curriculum_bounds(
            tf.cast(config.productivity_min, TENSORFLOW_DTYPE),
            tf.cast(config.productivity_max, TENSORFLOW_DTYPE),
            tf.cast(1.0, TENSORFLOW_DTYPE),
            progress,
            config.curriculum_initial_ratio
        )
        samples = StateSampler._sample_beta_scaled(
            shape=(batch_size,), minval=z_min, maxval=z_max
        )
        return tf.reshape(samples, (-1, 1))

    @staticmethod
    def _sample_debt(
        batch_size: tf.Tensor,
        config: DeepLearningConfig,
        progress: tf.Tensor
    ) -> tf.Tensor:
        """Sample debt values with curriculum bounds."""
        b_min, b_max = CurriculumBounds.get_curriculum_bounds(
            tf.cast(config.debt_min, TENSORFLOW_DTYPE),
            tf.cast(config.debt_max, TENSORFLOW_DTYPE),
            tf.cast(config.debt_min, TENSORFLOW_DTYPE),
            progress,
            config.curriculum_initial_ratio
        )
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