# econ_models/core/econ/debt_flow.py
"""
Cash flow calculations for equity valuation.

This module implements dividend and payout calculations for both
the basic model and the risky debt model with external finance costs.
"""

import tensorflow as tf

from econ_models.config.economic_params import EconomicParams
from econ_models.core.types import TENSORFLOW_DTYPE, Tensor


class DebtFlowCalculator:
    """Static methods for payout calculations."""

    @staticmethod
    def calculate(
        debt_next: Tensor,
        bond_price: Tensor,
        params: EconomicParams,
    ) -> Tensor:
        """
        Calculate equity cash flow including costly external finance.

        Formula:
            e = Profit + q*B' + TaxShield - I - Phi - B
            V = e - eta(e)

        Where eta(e) = (eta_0 + eta_1 * |e|) * 1{e < 0}

        Args:
            profit: After-tax profit.
            investment: Investment expenditure.
            adj_cost: Adjustment costs.
            debt_curr: Current debt (B).
            debt_next: Next period debt (B').
            bond_price: Bond price (q).
            params: Economic parameters.

        Returns:
            Final payout value to shareholders.
        """
        one = tf.cast(1.0, TENSORFLOW_DTYPE)
        tax_rate = tf.cast(params.corporate_tax_rate, TENSORFLOW_DTYPE)
        r_risk_free = tf.cast(params.risk_free_rate, TENSORFLOW_DTYPE)

        # Calculate raw payout
        debt_inflow = bond_price * debt_next

        interest_implied = (one - bond_price) * tf.maximum(
            debt_next, tf.cast(0.0, TENSORFLOW_DTYPE)
        )
        tax_shield = (tax_rate * interest_implied) / (one + r_risk_free)

        return debt_inflow, tax_shield