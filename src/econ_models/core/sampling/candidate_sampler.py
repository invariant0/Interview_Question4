# econ_models/core/sampling/candidate_sampler.py
"""
Candidate action sampling for optimization.

This module provides functions for generating candidate next-period
states during the policy optimization step of deep learning training.
"""

from typing import Tuple, Optional

import tensorflow as tf
import tensorflow_probability as tfp

from econ_models.config.dl_config import DeepLearningConfig
from econ_models.core.types import TENSORFLOW_DTYPE, Tensor
from econ_models.core.sampling.curriculum import CurriculumBounds

tfd = tfp.distributions


class CandidateSampler:
    """Utilities for sampling candidate next-period states."""

    @staticmethod
    def sample_candidate(
        batch_size: int,
        n_candidates: int,
        k_current: Tensor,
        b_current: Tensor,
        config: DeepLearningConfig,
        progress: Optional[tf.Tensor] = None,
        std_ratio_max: float = 0.4,
        std_ratio_min: float = 0.1,
        uniform_prob_initial: float = 1.0,
        uniform_prob_final: float = 0.0
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Generate candidates for next-period capital and debt.

        Uses a mixture of uniform and normal sampling that transitions
        from exploration (uniform) to exploitation (normal around current).

        Args:
            batch_size: Number of state points to evaluate.
            n_candidates: Number of candidates per state.
            k_current: Current capital values, shape (batch_size, 1).
            b_current: Current debt values, shape (batch_size, 1).
            config: Configuration with domain bounds.
            progress: Curriculum progress [0, 1].
            std_ratio_max: Normal std ratio at progress=0.
            std_ratio_min: Normal std ratio at progress=1.
            uniform_prob_initial: Uniform probability at progress=0.
            uniform_prob_final: Uniform probability at progress=1.

        Returns:
            Tuple of (k_cand, b_cand), each shape (batch_size, n_candidates).
        """
        progress = CandidateSampler._normalize_progress(progress)
        n_candidates = CandidateSampler._scale_candidates(n_candidates, progress)

        # Get curriculum bounds
        k_min, k_max = CandidateSampler._get_capital_bounds(config, progress)
        b_min, b_max = CandidateSampler._get_debt_bounds(config, progress)

        # Sample using mixture distribution
        k_cand = CandidateSampler._sample_mixture(
            batch_size, n_candidates, k_current,
            k_min, k_max, progress,
            std_ratio_max, std_ratio_min,
            uniform_prob_initial, uniform_prob_final
        )

        b_cand = CandidateSampler._sample_mixture(
            batch_size, n_candidates, b_current,
            b_min, b_max, progress,
            std_ratio_max, std_ratio_min,
            uniform_prob_initial, uniform_prob_final
        )

        return k_cand, b_cand

    @staticmethod
    def _normalize_progress(progress: Optional[tf.Tensor]) -> tf.Tensor:
        """Ensure progress is a valid tensor."""
        if progress is None:
            return tf.cast(1.0, TENSORFLOW_DTYPE)
        return tf.cast(progress, TENSORFLOW_DTYPE)

    @staticmethod
    def _scale_candidates(n_candidates: int, progress: tf.Tensor) -> tf.Tensor:
        """Scale number of candidates with progress."""
        min_batch_ratio = 0.1
        batch_ratio = min_batch_ratio + (1.0 - min_batch_ratio) * progress
        n_scaled = tf.cast(
            tf.cast(n_candidates, TENSORFLOW_DTYPE) * batch_ratio,
            tf.int32
        )
        return tf.maximum(n_scaled, 16)

    @staticmethod
    def _get_capital_bounds(
        config: DeepLearningConfig,
        progress: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Get curriculum-adjusted capital bounds."""
        return CurriculumBounds.get_curriculum_bounds(
            tf.cast(config.capital_min, TENSORFLOW_DTYPE),
            tf.cast(config.capital_max, TENSORFLOW_DTYPE),
            tf.cast(config.capital_steady_state, TENSORFLOW_DTYPE),
            progress,
            config.curriculum_initial_ratio
        )

    @staticmethod
    def _get_debt_bounds(
        config: DeepLearningConfig,
        progress: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Get curriculum-adjusted debt bounds."""
        return CurriculumBounds.get_curriculum_bounds(
            tf.cast(config.debt_min, TENSORFLOW_DTYPE),
            tf.cast(config.debt_max, TENSORFLOW_DTYPE),
            tf.cast(config.debt_min, TENSORFLOW_DTYPE),
            progress,
            config.curriculum_initial_ratio
        )

    @staticmethod
    def _sample_mixture(
        batch_size: int,
        n_candidates: tf.Tensor,
        current_values: Tensor,
        val_min: tf.Tensor,
        val_max: tf.Tensor,
        progress: tf.Tensor,
        std_ratio_max: float,
        std_ratio_min: float,
        uniform_prob_initial: float,
        uniform_prob_final: float
    ) -> tf.Tensor:
        """
        Sample from a mixture of uniform and normal distributions.

        Args:
            batch_size: Number of batch elements.
            n_candidates: Number of candidates per element.
            current_values: Current state values for centering normal.
            val_min: Minimum bound.
            val_max: Maximum bound.
            progress: Training progress [0, 1].
            std_ratio_max: Normal std ratio at progress=0.
            std_ratio_min: Normal std ratio at progress=1.
            uniform_prob_initial: Uniform probability at progress=0.
            uniform_prob_final: Uniform probability at progress=1.

        Returns:
            Sampled candidates, shape (batch_size, n_candidates).
        """
        val_range = val_max - val_min

        # Sample from uniform distribution
        uniform_samples = tfd.Uniform(
            low=val_min, high=val_max
        ).sample((batch_size, n_candidates))

        # Calculate adaptive std using cosine decay
        cosine_decay = 0.5 * (1.0 + tf.cos(3.14159 * progress))
        std_ratio = (
            tf.cast(std_ratio_min, TENSORFLOW_DTYPE)
            + (tf.cast(std_ratio_max, TENSORFLOW_DTYPE)
               - tf.cast(std_ratio_min, TENSORFLOW_DTYPE)) * cosine_decay
        )
        std = val_range * std_ratio

        # Sample from normal centered at current values
        center = tf.reshape(current_values, (batch_size, 1))
        normal_samples = center + std * tf.random.normal(
            shape=(batch_size, n_candidates),
            dtype=TENSORFLOW_DTYPE
        )
        normal_samples = tf.clip_by_value(normal_samples, val_min, val_max)

        # Calculate mixture probability
        uniform_prob = (
            tf.cast(uniform_prob_final, TENSORFLOW_DTYPE)
            + (tf.cast(uniform_prob_initial, TENSORFLOW_DTYPE)
               - tf.cast(uniform_prob_final, TENSORFLOW_DTYPE)) * cosine_decay
        )

        # Generate mixture mask
        random_vals = tf.random.uniform(
            shape=(batch_size, n_candidates),
            dtype=TENSORFLOW_DTYPE
        )
        use_uniform = random_vals < uniform_prob

        return tf.where(use_uniform, uniform_samples, normal_samples)