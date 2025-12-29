# econ_models/core/econ/bond_pricing.py
"""
Bond pricing and default-related calculations.

This module implements risk-neutral bond pricing with endogenous
default probabilities and recovery values.
"""

from typing import Optional

import tensorflow as tf

from econ_models.config.economic_params import EconomicParams
from econ_models.core.types import TENSORFLOW_DTYPE, Tensor


class BondPricingCalculator:
    """Static methods for bond pricing calculations."""

    @staticmethod
    def recovery_value(
        profit_next: Tensor,
        capital_next: Tensor,
        params: EconomicParams
    ) -> Tensor:
        """
        Calculate firm liquidation value in default.

        Formula: V_recovery = (1 - xi) * (Profit' + (1 - delta) * K')

        Args:
            profit_next: Operating profit in next period.
            capital_next: Capital stock in next period.
            params: Economic parameters (default cost, depreciation).

        Returns:
            Recoverable value for bondholders.
        """
        one = tf.cast(1.0, TENSORFLOW_DTYPE)
        default_cost = tf.cast(params.default_cost_proportional, TENSORFLOW_DTYPE)
        delta = tf.cast(params.depreciation_rate, TENSORFLOW_DTYPE)

        firm_value_gross = profit_next + (one - delta) * capital_next
        return (one - default_cost) * firm_value_gross

    @staticmethod
    def bond_payoff(
        recovery_val: Tensor,
        debt_face_value: Tensor,
        is_default: Tensor
    ) -> Tensor:
        """
        Determine ex-post payoff to bondholders.

        Args:
            recovery_val: Firm liquidation value.
            debt_face_value: Face value of debt claim.
            is_default: Indicator tensor (1.0 if default, 0.0 otherwise).

        Returns:
            Actual payoff received by bondholders.
        """
        b_face_safe = tf.maximum(debt_face_value, tf.cast(0.0, TENSORFLOW_DTYPE))

        # In default, bondholders receive recovery value capped at face value
        payoff_default = tf.minimum(recovery_val, b_face_safe)

        one = tf.cast(1.0, TENSORFLOW_DTYPE)
        return is_default * payoff_default + (one - is_default) * b_face_safe

    @staticmethod
    def risk_neutral_price(
        expected_payoff: Tensor,
        debt_next: Tensor,
        risk_free_rate: float,
        epsilon_debt: float,
        min_price: float,
        risk_free_price_val: Optional[float] = None
    ) -> Tensor:
        """
        Compute risk-neutral bond price.

        Formula: q = E[Payoff] / ((1 + r) * B')

        Args:
            expected_payoff: Expected future payment to bondholders.
            debt_next: Face value of new debt (B').
            risk_free_rate: Risk-free interest rate (r).
            epsilon_debt: Threshold for zero-debt detection.
            min_price: Minimum bond price floor.
            risk_free_price_val: Pre-computed 1/(1+r) for efficiency.

        Returns:
            Bond price schedule.
        """
        r_rf_t = tf.cast(risk_free_rate, TENSORFLOW_DTYPE)
        one = tf.cast(1.0, TENSORFLOW_DTYPE)

        if risk_free_price_val is None:
            q_rf = one / (one + r_rf_t)
        else:
            q_rf = tf.cast(risk_free_price_val, TENSORFLOW_DTYPE)

        b_eps_t = tf.cast(epsilon_debt, TENSORFLOW_DTYPE)
        q_min_t = tf.cast(min_price, TENSORFLOW_DTYPE)

        # Mask for states with non-trivial debt
        has_debt = debt_next > b_eps_t

        # Denominator: (1 + r) * B'
        denominator = (one + r_rf_t) * tf.maximum(debt_next, b_eps_t)
        q_risky = expected_payoff / denominator

        # Use risk-free price for zero debt, risky price otherwise
        q_final = tf.where(has_debt, q_risky, q_rf)

        return tf.clip_by_value(q_final, q_min_t, q_rf)