# econ_models/vfi/basic.py
"""
Basic RBC Model using Value Function Iteration.

This module implements a standard Real Business Cycle model with
capital adjustment costs, solved using discrete-grid VFI.

Example:
    >>> from econ_models.vfi.basic import BasicModelVFI
    >>> model = BasicModelVFI(params, config)
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
    CashFlowCalculator
)


class BasicModelVFI:
    """
    Solve standard RBC model with capital adjustment costs.

    State Space: (Capital K, Productivity Z)
    Choice Variable: (Capital K')

    Attributes:
        params: Economic parameters.
        config: Grid configuration.
        z_grid: Discretized productivity grid.
        k_grid: Discretized capital grid.
        P: Productivity transition matrix.
        k_ss: Steady state capital.
        z_min: Minimum productivity value.
        z_max: Maximum productivity value.
    """

    def __init__(
        self,
        params: EconomicParams,
        config: GridConfig,
        k_bounds: Optional[Tuple[float, float]] = None
    ) -> None:
        """
        Initialize the basic VFI model.

        Args:
            params: Economic parameters.
            config: Grid configuration.
            k_bounds: Optional custom bounds (min, max) for capital grid.
        """
        self.params = params
        self.config = config
        self.custom_k_bounds = k_bounds

        self._initialize_grids()
        self.vfi_engine = VFIEngine(
            params.discount_factor,
            self.P,
            config.tol_vfi,
            config.max_iter_vfi
        )

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
            default_scale=(0.1, 3.0)
        )

    def compute_flow(self) -> tf.Tensor:
        """
        Compute the 3D flow matrix: Flow[k_curr, k_next, z_curr].

        Returns:
            Flow matrix tensor of shape (n_capital, n_capital, n_productivity).
        """
        n_k = self.config.n_capital
        n_z = self.config.n_productivity

        # Broadcasting setup for 3D operations
        k_curr = tf.reshape(self.k_grid, (n_k, 1, 1))
        k_next = tf.reshape(self.k_grid, (1, n_k, 1))
        z_curr = tf.reshape(self.z_grid, (1, 1, n_z))

        # Economic calculations
        production = ProductionFunctions.cobb_douglas(k_curr, z_curr, self.params)
        profit = (1.0 - self.params.corporate_tax_rate) * production

        investment = ProductionFunctions.calculate_investment(
            k_curr, k_next, self.params
        )
        adj_cost, _ = AdjustmentCostCalculator.calculate(
            investment, k_curr, self.params
        )

        return CashFlowCalculator.basic_cash_flow(profit, investment, adj_cost)

    def get_policy_indices(self, value_function: tf.Tensor) -> tf.Tensor:
        """
        Calculate optimal k' index for every state (k, z).

        Args:
            value_function: Converged value function V.

        Returns:
            Tensor of indices [k_curr, z_curr] pointing to optimal k_next.
        """
        flow = self.compute_flow()
        ev = tf.matmul(value_function, self.P, transpose_b=True)
        rhs = flow + self.params.discount_factor * ev[None, :, :]
        policy_idx = tf.argmax(rhs, axis=1)
        return tf.cast(policy_idx, tf.int32)

    def simulate(
        self,
        value_function: Array,
        n_steps: int = 1000,
        n_batches: int = 1000,
        seed: int | None = None
    ) -> Tuple[SimulationHistory, Dict[str, float]]:
        """
        Simulate economy to check boundary hits.

        Args:
            value_function: Value function array.
            n_steps: Number of simulation periods per batch.
            n_batches: Number of independent simulation batches.
            seed: Random seed for reproducibility.

        Returns:
            Tuple of (SimulationHistory with complete trajectories, 
                     statistics dictionary with boundary hit percentages).
        """
        v_tensor = tf.constant(value_function, dtype=TENSORFLOW_DTYPE)
        policy_map = self.get_policy_indices(v_tensor).numpy()

        return Simulator.simulate_basic_model(
            policy_map,
            self.P.numpy(),
            self.config.n_capital,
            self.config.n_productivity,
            n_steps=n_steps,
            n_batches=n_batches,
            seed=seed
        )

    def solve(self) -> Dict[str, Array]:
        """
        Solve the model via Value Function Iteration.

        Returns:
            Dictionary containing V, K grid, and Z grid.
        """
        flow = self.compute_flow()
        n_k, n_z = self.config.n_capital, self.config.n_productivity
        v_init = tf.zeros((n_k, n_z), dtype=TENSORFLOW_DTYPE)

        v_star = self.vfi_engine.run_vfi(v_init, flow)

        return {
            "V": v_star.numpy(),
            "K": self.k_grid.numpy(),
            "Z": self.z_grid.numpy()
        }