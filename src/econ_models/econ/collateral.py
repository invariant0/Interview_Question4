# econ_models/core/econ/collateral.py
"""
Collateral constraint calculations.

This module implements borrowing constraints based on
collateralizable firm value.
"""

import tensorflow as tf

from econ_models.config.economic_params import EconomicParams
from econ_models.core.types import TENSORFLOW_DTYPE, Tensor


class CollateralCalculator:
    """Static methods for collateral constraint calculations."""

    @staticmethod
    def calculate_limit(
        k_next: Tensor,
        z_min: float,
        params: EconomicParams,
        recovery_fraction: float = 0.5
    ) -> Tensor:
        """
        Calculate maximum allowable debt based on collateral constraints.

        This constraint ensures debt does not exceed minimum recoverable
        firm value, preventing unbounded borrowing from tax shields.

        Constraint:
            b' <= (1 - tau) * z_min * (k')^theta + tau * delta * k' + s * k'

        Where:
            tau: Corporate tax rate
            z_min: Minimum productivity level
            theta: Capital share in production
            delta: Depreciation rate
            s: Recoverable fraction of capital in liquidation

        Args:
            k_next: Next period capital stock tensor (K').
            z_min: Minimum productivity grid value.
            params: Economic parameters.
            recovery_fraction: Fraction of capital recoverable (s).

        Returns:
            Maximum allowable debt for each capital level.
        """
        tau = tf.cast(params.corporate_tax_rate, TENSORFLOW_DTYPE)
        theta = tf.cast(params.capital_share, TENSORFLOW_DTYPE)
        delta = tf.cast(params.depreciation_rate, TENSORFLOW_DTYPE)
        z_min_val = tf.cast(z_min, TENSORFLOW_DTYPE)
        s_val = tf.cast(recovery_fraction, TENSORFLOW_DTYPE)

        term_1 = (1.0 - tau) * z_min_val * (k_next ** theta)
        term_2 = tau * delta * k_next
        term_3 = s_val * k_next

        return term_1 + term_2 + term_3