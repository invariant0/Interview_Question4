# econ_models/core/econ/issuance_costs.py
"""
Capital issuance cost calculations.

This module implements convex and fixed issuance costs for capital
investment decisions.
"""

from typing import Tuple

import tensorflow as tf

from econ_models.config.economic_params import EconomicParams
from econ_models.core.types import TENSORFLOW_DTYPE, Tensor


class IssuanceCostCalculator:
    """Static methods for issuance cost calculations."""

    @staticmethod
    def calculate(
        payout: Tensor,
        params: EconomicParams
    ) -> Tensor:
        """
        Calculate equity issuance cost for negative payouts.

        Args:
            payout: Raw payout amount (negative indicates issuance).
            params: Economic parameters with issuance cost coefficients.

        Returns:
            Equity issuance cost (zero if payout is non-negative).
        """
        eta_0 = tf.cast(params.equity_issuance_cost_fixed, TENSORFLOW_DTYPE)
        eta_1 = tf.cast(params.equity_issuance_cost_linear, TENSORFLOW_DTYPE)

        is_issuing = tf.cast(payout < 0.0, TENSORFLOW_DTYPE)
        return (eta_0 + eta_1 * tf.abs(payout)) * is_issuing
    
    @staticmethod
    def calculate_with_grad(
        payout: tf.Tensor,
        params: EconomicParams,
        sigmoid_k: tf.Tensor = None,
    ) -> tf.Tensor:
        """
        Calculate equity issuance cost using a Straight-Through Estimator (STE).
        
        Forward pass: Exact hard threshold (accuracy is preserved).
        Backward pass: Soft sigmoid gradient (differentiability is preserved).

        Args:
            payout: Raw payout amount (negative indicates issuance).
            params: Economic parameters with issuance cost coefficients.
            sigmoid_k: Steepness of the STE sigmoid. Annealed during training
                (small early → large late) so gradient signal progresses from
                broad exploration to precise threshold awareness. If ``None``,
                defaults to 1.0 for backward compatibility.
        """
        eta_0 = tf.cast(params.equity_issuance_cost_fixed, TENSORFLOW_DTYPE)
        eta_1 = tf.cast(params.equity_issuance_cost_linear, TENSORFLOW_DTYPE)

        # 1. The Hard Mask (Accurate Forward Pass)
        is_issuing_hard = tf.cast(payout < 0.0, TENSORFLOW_DTYPE)

        # 2. The Soft Mask (Differentiable Backward Pass)
        # k controls steepness — annealed from small (broad gradient) to
        # large (sharp, accurate threshold) during training.
        if sigmoid_k is None:
            sigmoid_k = tf.constant(1.0, dtype=TENSORFLOW_DTYPE)
        is_issuing_soft = tf.math.sigmoid(-sigmoid_k * payout)

        # 3. Straight-Through Estimator
        # Forward:  Soft + (Hard - Soft) = Hard  (accurate)
        # Backward: d(Soft)/dx + 0       = d(Soft)/dx  (differentiable)
        is_issuing = is_issuing_soft + tf.stop_gradient(is_issuing_hard - is_issuing_soft)

        # 4. Calculate Cost
        cost = (eta_0 + eta_1 * tf.abs(payout)) * is_issuing
        
        return cost

    @staticmethod
    def calculate_with_grad_dist(
        payout: tf.Tensor,
        eta_0: tf.Tensor,
        eta_1: tf.Tensor
    ) -> tf.Tensor:
        """
        Calculate equity issuance cost using STE with explicit eta0/eta1 tensors.

        Same as calculate_with_grad but takes eta0/eta1 as tensor arguments
        instead of from EconomicParams. Used for distributional training where
        these parameters vary per sample.

        Args:
            payout: Raw payout amount (negative indicates issuance).
            eta_0: Equity issuance fixed cost (tensor, per-sample).
            eta_1: Equity issuance linear cost (tensor, per-sample).

        Returns:
            Equity issuance cost.
        """
        # 1. Hard mask (accurate forward pass)
        is_issuing_hard = tf.cast(payout < 0.0, TENSORFLOW_DTYPE)

        # 2. Soft mask (differentiable backward pass)
        k = tf.constant(10.0, dtype=TENSORFLOW_DTYPE)
        is_issuing_soft = tf.math.sigmoid(-k * payout)

        # 3. STE: forward uses hard, backward uses soft
        is_issuing = is_issuing_soft + tf.stop_gradient(is_issuing_hard - is_issuing_soft)

        # 4. Calculate cost
        cost = (eta_0 + eta_1 * tf.abs(payout)) * is_issuing

        return cost