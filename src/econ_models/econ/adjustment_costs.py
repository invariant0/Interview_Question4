# econ_models/core/econ/adjustment_costs.py
"""
Capital adjustment cost calculations.

This module implements convex and fixed adjustment costs for capital
investment decisions.
"""

from typing import Tuple

import tensorflow as tf

from econ_models.config.economic_params import EconomicParams
from econ_models.core.types import TENSORFLOW_DTYPE, Tensor


class AdjustmentCostCalculator:
    """Static methods for adjustment cost calculations."""

    @staticmethod
    def calculate(
        investment: Tensor,
        capital_curr: Tensor,
        params: EconomicParams
    ) -> Tuple[Tensor, Tensor]:
        """
        Calculate convex and fixed capital adjustment costs.

        Cost Function:
            psi(I, K) = (psi_0 / 2) * (I^2 / K) + psi_1 * K * 1{I != 0}

        Args:
            investment: Investment amount (I).
            capital_curr: Current capital stock (K).
            params: Economic parameters with adjustment cost coefficients.

        Returns:
            Tuple containing:
                - Total adjustment cost (convex + fixed).
                - Marginal cost of the convex component (for Euler equations).
        """
        psi_0 = params.adjustment_cost_convex
        psi_1 = params.adjustment_cost_fixed

        # Prevent division by zero
        safe_capital = tf.maximum(capital_curr, 1e-4)

        # Convex cost component
        convex_cost = psi_0 * (investment ** 2) / (2.0 * safe_capital)

        # Fixed cost component (incurred if investment is non-zero)
        is_investing = tf.cast(tf.not_equal(investment, 0.0), TENSORFLOW_DTYPE)
        fixed_cost = psi_1 * capital_curr * is_investing

        total_cost = convex_cost + fixed_cost

        # Marginal cost for Euler equations (derivative of convex component)
        marginal_cost = psi_0 * investment / safe_capital

        return total_cost, marginal_cost