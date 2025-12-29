# econ_models/core/econ/cash_flow.py
"""
Cash flow calculations for equity valuation.

This module implements dividend and payout calculations for both
the basic model and the risky debt model with external finance costs.
"""

import tensorflow as tf

from econ_models.config.economic_params import EconomicParams
from econ_models.core.types import TENSORFLOW_DTYPE, Tensor


class CashFlowCalculator:
    """Static methods for cash flow calculations."""

    @staticmethod
    def basic_cash_flow(
        profit: Tensor,
        investment: Tensor,
        adj_cost: Tensor
    ) -> Tensor:
        """
        Compute standard equity cash flow (dividend).

        Formula: D = (1 - tau) * Y - I - Phi

        Args:
            profit: After-tax operating profit.
            investment: Investment expenditure.
            adj_cost: Capital adjustment costs.

        Returns:
            Net cash flow to shareholders.
        """
        return profit - investment - adj_cost

    @staticmethod
    def risky_cash_flow(
        revenue: Tensor,
        investment: Tensor,
        adj_cost: Tensor,
        debt_curr: Tensor,
        debt_next: Tensor,
        bond_price: Tensor,
        params: EconomicParams
    ) -> Tensor:
        """
        Calculate equity cash flow including costly external finance.

        Formula:
            e = Rev + q*B' + TaxShield - I - Phi - B
            V = e - eta(e)

        Where eta(e) = (eta_0 + eta_1 * |e|) * 1{e < 0}

        Args:
            revenue: After-tax revenue.
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

        e_raw = (
            revenue + debt_inflow + tax_shield
            - investment - adj_cost - debt_curr
        )

        # Calculate equity issuance cost
        issuance_cost = CashFlowCalculator._calculate_issuance_cost(e_raw, params)

        return e_raw - issuance_cost

    @staticmethod
    def _calculate_issuance_cost(
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