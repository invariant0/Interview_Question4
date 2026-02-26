# econ_models/core/econ/steady_state.py
"""
Steady state calculations for economic models.

This module computes analytical steady state values used for
grid construction and value function normalization.
"""

from econ_models.config.economic_params import EconomicParams
import tensorflow as tf

class SteadyStateCalculator:
    """Static methods for steady state calculations."""

    @staticmethod
    def calculate_capital(params: EconomicParams) -> float:
        """
        Calculate steady-state capital stock for the deterministic model.

        Derived from the Euler equation in steady state:
            k_ss = ((1 - tau) * theta / (r + delta))^(1 / (1 - theta))

        Args:
            params: Economic parameters containing discount factor,
                    depreciation, tax rate, and capital share.

        Returns:
            The steady-state capital stock.
        """
        r_implied = (1.0 / params.discount_factor) - 1.0
        denom = r_implied + params.depreciation_rate

        k_ss = (
            ((1.0 - params.corporate_tax_rate) * params.capital_share) / denom
        ) ** (1.0 / (1.0 - params.capital_share))

        return k_ss