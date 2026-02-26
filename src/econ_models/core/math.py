# econ_models/core/math.py
"""
Math utilities for deep learning training.

Contains:
    * FischerBurmeisterLoss — complementarity loss for value function constraints

Grid interpolation functions have been moved to
``econ_models.vfi.grids.grid_utils`` (``interp_2d_batch``, ``interp_3d_batch``,
``interp_1d_batch``, etc.) — all XLA-compiled.
"""

import tensorflow as tf

from econ_models.core.types import Tensor


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
