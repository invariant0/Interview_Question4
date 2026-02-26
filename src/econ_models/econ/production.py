# econ_models/core/econ/production.py
"""
Production function calculations.

This module implements production technology and investment calculations
for the economic models.
"""

import tensorflow as tf

from econ_models.config.economic_params import EconomicParams
from econ_models.core.types import Tensor


class ProductionFunctions:
    """Static methods for production-related calculations."""

    @staticmethod
    def cobb_douglas(
        capital: Tensor,
        productivity: Tensor,
        params: EconomicParams
    ) -> Tensor:
        """
        Compute output using Cobb-Douglas production technology.

        Formula: Y = Z * K^theta

        Args:
            capital: Capital stock tensor (K).
            productivity: Productivity shock tensor (Z).
            params: Economic parameters containing capital share.

        Returns:
            Gross production output tensor.
        """
        return productivity * (capital ** params.capital_share)

    @staticmethod
    def calculate_investment(
        capital_curr: Tensor,
        capital_next: Tensor,
        params: EconomicParams
    ) -> Tensor:
        """
        Calculate investment required for capital transition.

        Formula: I = K' - (1 - delta) * K

        Args:
            capital_curr: Current capital stock (K).
            capital_next: Next period capital stock (K').
            params: Economic parameters containing depreciation rate.

        Returns:
            Investment amount tensor.
        """
        return capital_next - (1.0 - params.depreciation_rate) * capital_curr