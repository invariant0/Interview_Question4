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
        Calculate convex and fixed capital adjustment costs for always investing situation.

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
        safe_capital = tf.maximum(capital_curr, 1e-8)
        is_investing = tf.cast(
            tf.not_equal(investment, 0),
            TENSORFLOW_DTYPE
        )
        # Convex cost component
        convex_cost = psi_0 * (investment ** 2) / (2.0 * safe_capital)

        fixed_cost = psi_1 * capital_curr * is_investing

        total_cost = convex_cost + fixed_cost 

        # Marginal cost for Euler equations (derivative of convex component)
        marginal_cost = psi_0 * investment / safe_capital

        return total_cost, marginal_cost
    @staticmethod
    def calculate_with_grad(
        investment: Tensor,
        capital_curr: Tensor,
        invest_prob: Tensor,
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
        # is_investing = tf.cast(tf.not_equal(investment, 0.0), TENSORFLOW_DTYPE)
        # invest_decision = tf.cast(investment_prob > 0.5, TENSORFLOW_DTYPE)
        # invest_decision = investment_prob + tf.stop_gradient(invest_decision - investment_prob)
        fixed_cost = psi_1 * capital_curr * invest_prob

        total_cost = convex_cost + fixed_cost 

        # Marginal cost for Euler equations (derivative of convex component)
        marginal_cost = psi_0 * investment / safe_capital

        return total_cost, marginal_cost
    @staticmethod
    def calculate_dist(
        investment: Tensor,
        capital_curr: Tensor,
        adjustment_cost_convex: Tensor,
        adjustment_cost_fixed: Tensor,
        invest_prob_ste: Tensor | None = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Calculate convex and fixed capital adjustment costs
        (distributional variant with per-sample parameters).

        When *invest_prob_ste* is provided the fixed-cost component is
        scaled by the Straight-Through Estimator of invest probability
        (gate mechanism) instead of the hard 1{I != 0} indicator.

        Cost Function:
            psi(I, K) = (psi_0 / 2) * (I^2 / K) + psi_1 * K * gate

        Args:
            investment: Investment amount (I).
            capital_curr: Current capital stock (K).
            adjustment_cost_convex: Convex adjustment cost parameter per sample.
            adjustment_cost_fixed: Fixed adjustment cost parameter per sample.
            invest_prob_ste: Optional STE of investment probability.  When
                supplied, multiplies the fixed cost instead of 1{I != 0}.

        Returns:
            Tuple containing:
                - Total adjustment cost (convex + fixed).
                - Marginal cost of the convex component (for Euler equations).
        """
        psi_0 = adjustment_cost_convex
        psi_1 = adjustment_cost_fixed

        # Prevent division by zero
        safe_capital = tf.maximum(capital_curr, 1e-8)

        # Convex cost component
        convex_cost = psi_0 * (investment ** 2) / (2.0 * safe_capital)

        # Fixed cost: gated by invest_prob_ste when available
        if invest_prob_ste is not None:
            fixed_cost = psi_1 * capital_curr * invest_prob_ste
        else:
            is_investing = tf.cast(
                tf.not_equal(investment, 0),
                TENSORFLOW_DTYPE
            )
            fixed_cost = psi_1 * capital_curr * is_investing

        total_cost = convex_cost + fixed_cost

        # Marginal cost for Euler equations (derivative of convex component)
        marginal_cost = psi_0 * investment / safe_capital

        return total_cost, marginal_cost

    @staticmethod
    def calculate_with_grad_dist(
        investment: Tensor,
        capital_curr: Tensor,
        invest_prob: Tensor,
        adjustment_cost_convex: Tensor,
        adjustment_cost_fixed: Tensor,
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
        psi_0 = adjustment_cost_convex
        psi_1 = adjustment_cost_fixed

        # Prevent division by zero
        safe_capital = tf.maximum(capital_curr, 1e-4)

        # Convex cost component
        convex_cost = psi_0 * (investment ** 2) / (2.0 * safe_capital)

        # Fixed cost component (incurred if investment is non-zero)
        # is_investing = tf.cast(tf.not_equal(investment, 0.0), TENSORFLOW_DTYPE)
        # invest_decision = tf.cast(investment_prob > 0.5, TENSORFLOW_DTYPE)
        # invest_decision = investment_prob + tf.stop_gradient(invest_decision - investment_prob)
        fixed_cost = psi_1 * capital_curr * invest_prob

        total_cost = convex_cost + fixed_cost 

        # Marginal cost for Euler equations (derivative of convex component)
        marginal_cost = psi_0 * investment / safe_capital

        return total_cost, marginal_cost
    
    # @staticmethod
    # def calculate_with_grad(
    #     investment: Tensor,
    #     capital_curr: Tensor,
    #     params: EconomicParams
    # ) -> Tuple[Tensor, Tensor]:
    #     """
    #     Uses an L1-style proxy to encourage sparsity (inaction) during training.
    #     """
    #     psi_0 = params.adjustment_cost_convex
    #     psi_1 = params.adjustment_cost_fixed
    #     safe_capital = tf.maximum(capital_curr, 1e-4)

    #     # 1. Convex Cost (Standard)
    #     convex_cost = psi_0 * (tf.square(investment)) / (2.0 * safe_capital)
    #     marginal_convex = psi_0 * investment / safe_capital

    #     # 2. Fixed Cost Proxy for Gradients
    #     # Instead of psi_1 * 1{I!=0}, we use a proxy: lambda * |I|
    #     # This acts like a linear transaction cost.
    #     # If lambda is high enough, the optimal action is exactly 0.
        
    #     # We scale the proxy so it roughly matches the magnitude of the fixed cost
    #     # at a typical investment size, or treat it as a hyperparameter.
        
    #     # The gradient of |I| is sign(I).
    #     # This provides a CONSTANT "push" towards zero, unlike the tanh function
    #     # which provides zero push when I is large.
        
    #     proxy_weight = psi_1 * 0.1 # This is a hyperparameter you must tune!
    #     marginal_fixed_proxy = proxy_weight * tf.sign(investment)

    #     # 3. True Cost (For reporting/logging only)
    #     is_investing = tf.cast(tf.not_equal(investment, 0.0), TENSORFLOW_DTYPE)
    #     true_fixed_cost = psi_1 * capital_curr * is_investing
    #     total_true_cost = convex_cost + true_fixed_cost

    #     # 4. Effective Marginal Cost
    #     # This is what the Euler equation will "see".
    #     total_marginal_cost = marginal_convex + marginal_fixed_proxy

    #     return total_true_cost, total_marginal_cost