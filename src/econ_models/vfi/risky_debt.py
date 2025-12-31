# econ_models/vfi/risky_debt.py
"""
Risky Debt Model using Value Function Iteration.

This module implements a corporate finance model with risky debt
and endogenous default, solved using discrete-grid VFI with
an outer loop for bond price iteration.

Example:
    >>> from econ_models.vfi.risky_debt import RiskyDebtModelVFI
    >>> model = RiskyDebtModelVFI(params, config)
    >>> results = model.solve()
"""

from typing import Dict, Optional, Tuple

import tensorflow as tf
import numpy as np

from econ_models.config.economic_params import EconomicParams
from econ_models.config.vfi_config import GridConfig
from econ_models.vfi.engine import VFIEngine
from econ_models.vfi.grids.grid_builder import GridBuilder
from econ_models.vfi.simulation.simulator import Simulator, SimulationHistory
from econ_models.core.types import TENSORFLOW_DTYPE, Array
from econ_models.econ import (
    ProductionFunctions,
    AdjustmentCostCalculator,
    CashFlowCalculator,
    BondPricingCalculator,
    CollateralCalculator
)


class RiskyDebtModelVFI:
    """
    Solve corporate finance model with risky debt and default options.

    State Space: (Capital K, Debt B, Productivity Z)
    Choice Variables: (Capital K', Debt B')

    The model is solved using an outer loop that iterates on bond prices
    and an inner loop that performs standard VFI given those prices.

    Attributes:
        params: Economic parameters.
        config: Grid configuration.
        z_grid: Discretized productivity grid.
        k_grid: Discretized capital grid.
        b_grid: Discretized debt grid.
        P: Productivity transition matrix.
        bond_prices: Current bond price schedule.
    """

    def __init__(
        self,
        params: EconomicParams,
        config: GridConfig,
        k_bounds: Optional[Tuple[float, float]] = None,
        b_bounds: Optional[Tuple[float, float]] = None
    ) -> None:
        """
        Initialize the risky debt VFI model.

        Args:
            params: Economic parameters.
            config: Grid configuration.
            k_bounds: Optional custom bounds (min, max) for capital grid.
            b_bounds: Optional custom bounds (min, max) for debt grid.

        Raises:
            ValueError: If n_debt < 2 in configuration.
        """
        self.params = params
        self.config = config
        self.custom_k_bounds = k_bounds
        self.custom_b_bounds = b_bounds

        self._validate_config()
        self._initialize_grids()

        self.vfi_engine = VFIEngine(
            params.discount_factor,
            self.P,
            config.tol_vfi,
            config.max_iter_vfi
        )

    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        if self.config.n_debt < 2:
            raise ValueError("RiskyDebtModel requires GridConfig.n_debt >= 2.")

    def _initialize_grids(self) -> None:
        """Set up discretized state space grids."""
        # Productivity grid
        self.z_grid, self.P, self.z_min, self.z_max = (
            GridBuilder.build_productivity_grid(self.config, self.params)
        )

        # Capital grid
        self.k_grid, self.k_ss = GridBuilder.build_capital_grid(
            self.config,
            self.params,
            custom_bounds=self.custom_k_bounds,
            default_scale=(0.1, 10.0)
        )

        # Debt grid
        self.b_grid = GridBuilder.build_debt_grid(
            self.config,
            self.k_ss,
            custom_bounds=self.custom_b_bounds
        )

        # Initialize bond prices at risk-free level
        shape = (
            self.config.n_capital,
            self.config.n_debt,
            self.config.n_productivity
        )
        self.bond_prices = GridBuilder.build_initial_bond_prices(
            shape, self.params.risk_free_rate
        )

    def compute_flow(
        self,
        q_sched: tf.Tensor,
        enforce_constraint: bool = False
    ) -> tf.Tensor:
        """
        Compute the 5D flow matrix: Flow[k, b, k', b', z].

        Args:
            q_sched: Bond price schedule tensor.
            enforce_constraint: If True, apply collateral constraint.

        Returns:
            Flow tensor of shape (nk, nb, nk, nb, nz).
        """
        n_k = self.config.n_capital
        n_b = self.config.n_debt
        n_z = self.config.n_productivity

        # Broadcasting setup for 5D operations
        k_curr = tf.reshape(self.k_grid, (n_k, 1, 1, 1, 1))
        k_next = tf.reshape(self.k_grid, (1, 1, n_k, 1, 1))
        b_curr = tf.reshape(self.b_grid, (1, n_b, 1, 1, 1))
        b_next = tf.reshape(self.b_grid, (1, 1, 1, n_b, 1))
        z_curr = tf.reshape(self.z_grid, (1, 1, 1, 1, n_z))

        # Economic calculations
        production = ProductionFunctions.cobb_douglas(k_curr, z_curr, self.params)
        revenue = (1.0 - self.params.corporate_tax_rate) * production

        investment = ProductionFunctions.calculate_investment(
            k_curr, k_next, self.params
        )
        adj_cost, _ = AdjustmentCostCalculator.calculate(
            investment, k_curr, self.params
        )

        flow = CashFlowCalculator.risky_cash_flow(
            revenue, investment, adj_cost,
            b_curr, b_next, q_sched, self.params
        )

        if enforce_constraint:
            flow = self._apply_collateral_constraint(flow, k_next, b_next)

        return flow

    def _apply_collateral_constraint(
        self,
        flow: tf.Tensor,
        k_next: tf.Tensor,
        b_next: tf.Tensor
    ) -> tf.Tensor:
        """Apply collateral constraint penalty to invalid debt levels."""
        collateral_limit = CollateralCalculator.calculate_limit(
            k_next,
            self.z_min,
            self.params,
            self.params.collateral_recovery_fraction
        )

        valid_mask = b_next <= collateral_limit
        penalty = tf.cast(self.config.collateral_violation_penalty, TENSORFLOW_DTYPE)

        return tf.where(valid_mask, flow, penalty)

    def _update_prices(self, value_function: tf.Tensor) -> float:
        """
        Update bond prices based on default probabilities.

        Args:
            value_function: Converged value function V(k, b, z).

        Returns:
            Maximum absolute difference for convergence checking.
        """
        n_k = self.config.n_capital
        n_b = self.config.n_debt
        n_z = self.config.n_productivity

        # Determine default set
        is_default = tf.cast(
            value_function <= self.config.v_default_eps,
            TENSORFLOW_DTYPE
        )

        # Calculate recovery values
        k_next_col = tf.reshape(self.k_grid, (n_k, 1, 1))
        z_next_col = tf.reshape(self.z_grid, (1, 1, n_z))

        profit_next = (
            (1.0 - self.params.corporate_tax_rate)
            * ProductionFunctions.cobb_douglas(k_next_col, z_next_col, self.params)
        )
        recovery_val = BondPricingCalculator.recovery_value(
            profit_next, k_next_col, self.params
        )

        # Calculate bond payoffs
        b_next_col = tf.reshape(self.b_grid, (1, n_b, 1))
        payoff_next = BondPricingCalculator.bond_payoff(
            recovery_val, b_next_col, is_default
        )

        # Expected payoff via integration over Z
        payoff_flat = tf.reshape(payoff_next, (-1, n_z))
        exp_payoff_flat = tf.matmul(payoff_flat, self.P, transpose_b=True)
        exp_payoff = tf.reshape(exp_payoff_flat, (n_k, n_b, n_z))

        # Risk-neutral pricing
        r_rf = tf.cast(self.params.risk_free_rate, TENSORFLOW_DTYPE)
        q_rf_val = 1.0 / (1.0 + r_rf)

        q_new = BondPricingCalculator.risk_neutral_price(
            expected_payoff=exp_payoff,
            debt_next=b_next_col,
            risk_free_rate=self.params.risk_free_rate,
            epsilon_debt=self.config.b_eps,
            min_price=self.config.q_min,
            risk_free_price_val=float(q_rf_val)
        )

        # Update with relaxation
        diff = tf.reduce_max(tf.abs(q_new - self.bond_prices))
        lam = tf.cast(self.config.relax_q, TENSORFLOW_DTYPE)
        self.bond_prices = lam * q_new + (1.0 - lam) * self.bond_prices

        return float(diff.numpy())

    def get_policy_indices(
        self,
        value_function: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Extract optimal policy indices for capital and debt.

        Args:
            value_function: Converged value function V.

        Returns:
            Tuple of (k' indices, b' indices).
        """
        n_k = self.config.n_capital
        n_b = self.config.n_debt
        n_z = self.config.n_productivity

        q_sched = tf.reshape(self.bond_prices, (1, 1, n_k, n_b, n_z))
        flow = self.compute_flow(q_sched, enforce_constraint=False)

        ev = tf.tensordot(value_function, self.P, axes=[[2], [1]])
        rhs = flow + self.params.discount_factor * ev[None, None, :, :, :]

        # Flatten choice dims for argmax
        rhs_flat = tf.reshape(rhs, (n_k, n_b, n_k * n_b, n_z))
        best_idx_flat = tf.argmax(rhs_flat, axis=2)

        # Unravel indices
        best_k_idx = best_idx_flat // n_b
        best_b_idx = best_idx_flat % n_b

        return tf.cast(best_k_idx, tf.int32), tf.cast(best_b_idx, tf.int32)

    def simulate(
        self,
        value_function: Array,
        n_steps: int = 1000,
        n_batches: int = 1000,
        seed: int | None = None
    ) -> Tuple[SimulationHistory, Dict[str, float]]:
        """
        Simulate economy to check grid boundary hits.

        Args:
            value_function: Converged value function array.
            n_steps: Number of simulation periods per batch.
            n_batches: Number of independent simulation batches.
            seed: Random seed for reproducibility.

        Returns:
            Tuple of (SimulationHistory with complete trajectories,
                     statistics dictionary with boundary hit percentages).
        """
        v_tensor = tf.constant(value_function, dtype=TENSORFLOW_DTYPE)
        pol_k, pol_b = self.get_policy_indices(v_tensor)

        return Simulator.simulate_risky_model(
            pol_k.numpy(),
            pol_b.numpy(),
            self.P.numpy(),
            self.config.n_capital,
            self.config.n_debt,
            self.config.n_productivity,
            n_steps=n_steps,
            n_batches=n_batches,
            seed=seed
        )

    def solve(
        self,
        use_chunked: bool = False,
        chunk_size_k: Optional[int] = None,
        chunk_size_b: Optional[int] = None
    ) -> Dict[str, Array]:
        """
        Solve full equilibrium by iterating on bond prices and values.

        Args:
            use_chunked: If True, use memory-efficient chunked VFI.
            chunk_size_k: Chunk size for capital dimension.
            chunk_size_b: Chunk size for debt dimension.

        Returns:
            Dictionary containing V, Q, K, B, Z arrays.

        Raises:
            ValueError: If grid sizes exceed memory limits.
        """
        n_k = self.config.n_capital
        n_b = self.config.n_debt
        n_z = self.config.n_productivity

        self._validate_grid_size(n_k, n_b)

        value_function = tf.zeros((n_k, n_b, n_z), dtype=TENSORFLOW_DTYPE)

        for outer_iter in range(self.config.max_outer):
            is_risk_free_iter = (outer_iter == 0)
            v_old = value_function

            # Inner VFI loop
            q_sched = tf.reshape(self.bond_prices, (1, 1, n_k, n_b, n_z))
            flow = self.compute_flow(q_sched, enforce_constraint=is_risk_free_iter)
            value_function = self.vfi_engine.run_vfi(value_function, flow)

            # Update bond prices
            err_q = self._update_prices(value_function)

            # Check convergence
            err_v = tf.reduce_max(tf.abs(value_function - v_old))
            print(f"Outer Iter {outer_iter}: Err V = {err_v:.6f}, Err Q = {err_q:.6f}")

            if err_v < self.config.tol_outer and outer_iter >= 2:
                break

        return {
            "V": value_function.numpy(),
            "Q": self.bond_prices.numpy(),
            "K": self.k_grid.numpy(),
            "B": self.b_grid.numpy(),
            "Z": self.z_grid.numpy()
        }

    def _validate_grid_size(self, n_k: int, n_b: int) -> None:
        """Validate grid sizes for memory constraints."""
        if n_b > 100 or n_k > 100:
            raise ValueError(
                "Please decrease n_b or n_k to <= 100 for memory constraints."
            )