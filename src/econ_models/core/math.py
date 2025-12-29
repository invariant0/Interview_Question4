# econ_models/core/math.py
"""
Backward compatibility wrapper for math utilities.

This module maintains the original MathUtils interface while
delegating to the refactored sampling modules.
"""

from typing import Tuple, Optional

import tensorflow as tf

from econ_models.config.dl_config import DeepLearningConfig
from econ_models.core.types import TENSORFLOW_DTYPE, Tensor
from econ_models.core.sampling.transitions import TransitionFunctions
from econ_models.core.sampling.curriculum import CurriculumBounds
from econ_models.core.sampling.state_sampler import StateSampler
from econ_models.core.sampling.candidate_sampler import CandidateSampler


class MathUtils:
    """
    Backward-compatible wrapper for mathematical utilities.

    This class delegates to specialized modules while maintaining
    the original API for existing code.
    """

    log_ar1_transition = staticmethod(TransitionFunctions.log_ar1_transition)
    get_curriculum_bounds = staticmethod(CurriculumBounds.get_curriculum_bounds)
    sample_states = staticmethod(StateSampler.sample_states)
    sample_candidate = staticmethod(CandidateSampler.sample_candidate)

    @staticmethod
    def sample_heavy_tailed_vector(
        shape: Tuple[int],
        minval: Tensor,
        maxval: Tensor,
        alpha: float = 1.0,
        beta: float = 1.0
    ) -> Tensor:
        """
        Generate samples from a scaled Beta distribution.

        Delegates to StateSampler._sample_beta_scaled.

        Args:
            shape: Output shape.
            minval: Minimum value.
            maxval: Maximum value.
            alpha: Beta distribution alpha parameter.
            beta: Beta distribution beta parameter.

        Returns:
            Samples scaled to [minval, maxval].
        """
        return StateSampler._sample_beta_scaled(
            shape, minval, maxval, alpha, beta
        )