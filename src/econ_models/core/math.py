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
    
class FischerBurmeisterLoss:
    """
    Fischer-Burmeister complementarity function for value function constraints.
    
    Implements the FB function: Φ(a, b) = a + b - sqrt(a^2 + b^2)
    
    The economic logic requires V(s) = max(0, Vcont(s)), which implies:
    1. Non-negativity: V(s) >= 0
    2. Dominance: V(s) >= Vcont(s)
    3. Complementarity: If V(s) > 0, then V(s) = Vcont(s)
    
    We enforce this using: Φ(V, V - Vcont) = 0
    
    For unbiased gradient estimation with stochastic Vcont, we use the
    two-shock approach where Vcont is estimated with two independent samples.
    """
    
    @staticmethod
    @tf.function
    def fb_function(a: tf.Tensor, b: tf.Tensor, epsilon: float = 1e-8) -> tf.Tensor:
        """
        Compute Fischer-Burmeister function Φ(a, b) = a + b - sqrt(a^2 + b^2).
        
        Args:
            a: First argument tensor.
            b: Second argument tensor.
            epsilon: Small constant for numerical stability.
            
        Returns:
            FB function values.
        """
        return a + b - tf.sqrt(a**2 + b**2 + epsilon)
    
    @staticmethod
    @tf.function
    def complementarity_residual(
        v_value: tf.Tensor,
        v_cont: tf.Tensor,
        epsilon: float = 1e-8
    ) -> tf.Tensor:
        """
        Compute FB residual for value function complementarity.
        
        Uses: Φ(V, V - Vcont) = 0
        
        Args:
            v_value: Value network output V(s).
            v_cont: Continuous network output Vcont(s).
            epsilon: Numerical stability constant.
            
        Returns:
            FB residual tensor.
        """
        gap = v_value - v_cont
        return FischerBurmeisterLoss.fb_function(v_value, gap, epsilon)
    
    @staticmethod
    @tf.function
    def compute_loss(
        v_value: tf.Tensor,
        v_cont: tf.Tensor,
        epsilon: float = 1e-8
    ) -> tf.Tensor:
        """
        Compute squared FB loss for training Value Net.
        
        Loss = E[(V + (V - Vcont) - sqrt(V^2 + (V - Vcont)^2))^2]
        
        Args:
            v_value: Value network output V(s).
            v_cont: Continuous network output Vcont(s).
            epsilon: Numerical stability constant.
            
        Returns:
            Mean squared FB residual.
        """
        residual = FischerBurmeisterLoss.complementarity_residual(
            v_value, v_cont, epsilon
        )
        return tf.reduce_mean(residual**2)