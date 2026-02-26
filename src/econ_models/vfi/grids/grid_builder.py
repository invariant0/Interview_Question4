# econ_models/vfi/grids/grid_builder.py
"""
Grid construction utilities for VFI state spaces.

This module provides functions for building capital, debt, and
productivity grids with appropriate spacing (linear or logarithmic).
"""

from typing import Tuple

import tensorflow as tf

from econ_models.config.economic_params import EconomicParams
from econ_models.config.vfi_config import GridConfig
from econ_models.core.types import TENSORFLOW_DTYPE, Tensor
from econ_models.econ import SteadyStateCalculator
from econ_models.vfi.grids.grid_utils import tauchen_discretization


class GridBuilder:
    """
    Utility class for constructing VFI state space grids.

    This class provides static methods for building discretized grids
    for capital, debt, and productivity state variables.
    """

    @staticmethod
    def build_productivity_grid(
        config: GridConfig,
        params: EconomicParams
    ) -> Tuple[Tensor, Tensor, float, float]:
        """
        Build productivity grid using Tauchen's method.

        Args:
            config: Grid configuration.
            params: Economic parameters with AR(1) process parameters.

        Returns:
            Tuple containing:
                - z_grid: Productivity grid tensor.
                - P: Transition probability matrix.
                - z_min: Minimum productivity value.
                - z_max: Maximum productivity value.
        """
        z_grid, P = tauchen_discretization(
            config.n_productivity,
            params.productivity_persistence,
            params.productivity_std_dev,
            config.tauchen_width
        )
        z_min = float(tf.reduce_min(z_grid))
        z_max = float(tf.reduce_max(z_grid))

        return z_grid, P, z_min, z_max

    @staticmethod
    def build_capital_grid(
        config: GridConfig,
        params: EconomicParams,
        custom_bounds: Tuple[float, float],
        use_log_spacing: bool = False,
    ) -> Tuple[Tensor, float]:
        """
        Build capital grid with optional log spacing.

        Args:
            config: Grid configuration.
            params: Economic parameters.
            custom_bounds: Custom (min, max) bounds.
            use_log_spacing: Whether to use logarithmic spacing.

        Returns:
            Tuple containing:
                - k_grid: Capital grid tensor.
                - k_ss: Steady state capital value.
        """
        k_ss = SteadyStateCalculator.calculate_capital(params)

        k_min_val, k_max_val = custom_bounds

        if use_log_spacing:
            k_grid = GridBuilder._build_log_spaced_grid(
                k_min_val, k_max_val, config.n_capital
            )
        else:
            k_grid = GridBuilder._build_linear_grid(
                k_min_val, k_max_val, config.n_capital
            )

        return k_grid, k_ss

    @staticmethod
    def build_debt_grid(
        config: GridConfig,
        k_ss: float,
        custom_bounds: Tuple[float, float],
    ) -> Tensor:
        """
        Build debt grid with linear spacing.

        Args:
            config: Grid configuration.
            k_ss: Steady state capital for default scaling.
            custom_bounds: Custom (min, max) bounds.

        Returns:
            Debt grid tensor.
        """
        b_min_val, b_max_val = custom_bounds
        return GridBuilder._build_linear_grid(
            b_min_val, b_max_val, config.n_debt
        )

    @staticmethod
    def build_initial_bond_prices(
        shape: Tuple[int, ...],
        risk_free_rate: float
    ) -> Tensor:
        """
        Initialize bond prices at risk-free level.

        Args:
            shape: Shape of the bond price tensor.
            risk_free_rate: Risk-free interest rate.

        Returns:
            Tensor of risk-free bond prices.
        """
        q_rf = 1.0 / (1.0 + risk_free_rate)
        return tf.fill(shape, tf.cast(q_rf, TENSORFLOW_DTYPE))

    @staticmethod
    def _build_log_spaced_grid(
        min_val: float,
        max_val: float,
        n_points: int
    ) -> Tensor:
        """Build a logarithmically-spaced grid."""
        # Ensure strict positivity for log to prevent NaNs if min_val=0
        safe_min = max(min_val, 1e-4)

        log_min = tf.math.log(tf.cast(safe_min, TENSORFLOW_DTYPE))
        log_max = tf.math.log(tf.cast(max_val, TENSORFLOW_DTYPE))
        log_grid = tf.linspace(log_min, log_max, n_points)
        return tf.exp(tf.cast(log_grid, TENSORFLOW_DTYPE))

    @staticmethod
    def _build_linear_grid(
        min_val: float,
        max_val: float,
        n_points: int
    ) -> Tensor:
        """Build a linearly-spaced grid."""
        return tf.cast(
            tf.linspace(min_val, max_val, n_points),
            TENSORFLOW_DTYPE
        )