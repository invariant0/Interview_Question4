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
        payout: Tensor,
        issuance_cost: Tensor,
    ) -> Tensor:
        """
        Compute equity cash flow including costly external finance.
        """
        return payout - issuance_cost