# econ_models/core/sampling/curriculum.py
"""
Curriculum learning utilities for progressive training.

This module provides functions for gradually expanding the training
domain from a safe region near steady state to the full state space.
"""

from typing import Tuple

import tensorflow as tf

from econ_models.core.types import TENSORFLOW_DTYPE, Tensor


class CurriculumBounds:
    """Utilities for curriculum-based boundary expansion."""

    @staticmethod
    def get_curriculum_bounds(
        global_min: Tensor,
        global_max: Tensor,
        safe_center: Tensor,
        progress: Tensor,
        initial_ratio: float
    ) -> Tuple[Tensor, Tensor]:
        """
        Calculate dynamic boundaries that expand from center to global bounds.

        As training progresses, the boundaries gradually expand from a small
        region around the safe center to the full global domain.

        Formula:
            ratio = initial_ratio + (1 - initial_ratio) * progress
            curr_min = center - (center - global_min) * ratio
            curr_max = center + (global_max - center) * ratio

        Args:
            global_min: Ultimate minimum bound.
            global_max: Ultimate maximum bound.
            safe_center: Center point for initial sampling.
            progress: Training progress in [0, 1].
            initial_ratio: Starting ratio of full domain (e.g., 0.05 = 5%).

        Returns:
            Tuple of current (min, max) bounds.
        """
        ratio = initial_ratio + (1.0 - initial_ratio) * progress

        curr_min = safe_center - (safe_center - global_min) * ratio
        curr_max = safe_center + (global_max - safe_center) * ratio

        return curr_min, curr_max