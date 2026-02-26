"""Value Function Iteration for the Basic RBC Model.

Solves a discrete-time capital-accumulation problem with fixed and convex
adjustment costs.  The firm each period chooses between:

* **ADJUST** — pay adjustment costs and choose optimal next-period capital K'.
* **WAIT**   — no investment; K' = (1 − δ) K.

Outputs both discrete grid indices (for backward compatibility) and
continuous policy values (for interpolation-based simulation).

Architecture note
-----------------
This module is a thin orchestrator.  Bellman-iteration primitives are
delegated to ``vfi.kernels.bellman_kernels``, and policy extraction to
``vfi.policies``.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import tensorflow as tf

from econ_models.config.economic_params import EconomicParams
from econ_models.config.vfi_config import GridConfig
from econ_models.core.types import TENSORFLOW_DTYPE, Array
from econ_models.econ import (
    AdjustmentCostCalculator,
    CashFlowCalculator,
    ProductionFunctions,
)
from econ_models.vfi.grids.grid_builder import GridBuilder
from econ_models.vfi.grids.grid_utils import interp_1d_batch
from econ_models.vfi.policies import extract_basic_policies

logger = logging.getLogger(__name__)


class BasicModelVFI:
    """VFI solver for the basic RBC model with adjustment costs.

    State space : (Capital K, Productivity Z)
    Choice      : Next-period capital K'

    Parameters
    ----------
    params : EconomicParams
        Structural economic parameters (frozen dataclass).
    config : GridConfig
        Grid sizes, tolerances, and iteration limits.
    k_bounds : tuple of (float, float), optional
        Custom ``(k_min, k_max)`` bounds for the capital grid.
        When *None*, ``GridBuilder`` uses its own default bounds.

    Raises
    ------
    ValueError
        If *k_bounds* is provided but contains invalid values.
    """

    def __init__(
        self,
        params: EconomicParams,
        config: GridConfig,
        k_bounds: Optional[Tuple[float, float]] = None,
    ) -> None:
        self._validate_inputs(params, config, k_bounds)

        self.params: EconomicParams = params
        self.config: GridConfig = config
        self.custom_k_bounds: Optional[Tuple[float, float]] = k_bounds

        self._initialize_grids()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_inputs(
        params: EconomicParams,
        config: GridConfig,
        k_bounds: Optional[Tuple[float, float]],
    ) -> None:
        """Validate constructor arguments.

        Raises
        ------
        ValueError
            On any invalid parameter combination.
        """
        if config.n_capital < 2:
            raise ValueError(
                f"n_capital must be >= 2, got {config.n_capital}."
            )
        if k_bounds is not None:
            k_lo, k_hi = k_bounds
            if k_lo >= k_hi:
                raise ValueError(
                    f"k_bounds lower ({k_lo}) must be less than upper ({k_hi})."
                )
            if k_lo <= 0.0:
                raise ValueError(
                    f"k_bounds lower must be positive, got {k_lo}."
                )

    # ------------------------------------------------------------------
    # Grid initialisation
    # ------------------------------------------------------------------

    def _initialize_grids(self) -> None:
        """Build discretised productivity and capital grids.

        Populates instance attributes consumed by flow-computation
        methods and the Bellman iteration loop.
        """
        self.z_grid: tf.Tensor
        self.P: tf.Tensor
        self.z_min: float
        self.z_max: float
        self.z_grid, self.P, self.z_min, self.z_max = (
            GridBuilder.build_productivity_grid(self.config, self.params)
        )

        self.k_grid: tf.Tensor
        self.k_ss: float
        self.k_grid, self.k_ss = GridBuilder.build_capital_grid(
            self.config,
            self.params,
            self.custom_k_bounds,
            use_log_spacing=False,
        )

        self.n_capital: int = int(tf.shape(self.k_grid)[0])
        self.n_productivity: int = self.config.n_productivity

        # Grid extremes — used by simulators for clamping.
        self.k_min: float = float(self.k_grid[0])
        self.k_max: float = float(self.k_grid[-1])

    # ------------------------------------------------------------------
    # Flow utilities
    # ------------------------------------------------------------------

    def compute_flow_adjust(self) -> tf.Tensor:
        """Compute the flow utility when the firm **adjusts** capital.

        Returns
        -------
        tf.Tensor
            Shape ``(n_k, n_k, n_z)`` — flow value for every
            ``(k_current, k_next, z)`` triple.
        """
        n_k = self.n_capital
        n_z = self.n_productivity

        k_curr = tf.reshape(self.k_grid, (n_k, 1, 1))
        k_next = tf.reshape(self.k_grid, (1, n_k, 1))
        z_curr = tf.reshape(self.z_grid, (1, 1, n_z))

        production = ProductionFunctions.cobb_douglas(
            k_curr, z_curr, self.params
        )
        profit = (1.0 - self.params.corporate_tax_rate) * production
        investment = ProductionFunctions.calculate_investment(
            k_curr, k_next, self.params
        )
        adjustment_cost, _ = AdjustmentCostCalculator.calculate(
            investment, k_curr, self.params
        )
        return CashFlowCalculator.basic_cash_flow(
            profit, investment, adjustment_cost
        )

    def compute_flow_wait(self) -> tf.Tensor:
        """Compute the flow utility when the firm **waits** (zero investment).

        Returns
        -------
        tf.Tensor
            Shape ``(n_k, n_z)`` — after-tax profit for every ``(k, z)`` pair.
        """
        n_k = self.n_capital
        n_z = self.n_productivity

        k_curr = tf.reshape(self.k_grid, (n_k, 1))
        z_curr = tf.reshape(self.z_grid, (1, n_z))

        production = ProductionFunctions.cobb_douglas(
            k_curr, z_curr, self.params
        )
        return (1.0 - self.params.corporate_tax_rate) * production

    # ------------------------------------------------------------------
    # Bellman iteration
    # ------------------------------------------------------------------

    def _run_bellman_iteration(
        self,
        flow_adjust: tf.Tensor,
        flow_wait: tf.Tensor,
        k_depreciated: tf.Tensor,
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """Iterate the Bellman equation until convergence.

        Parameters
        ----------
        flow_adjust : tf.Tensor
            Pre-computed ADJUST branch flow, shape ``(n_k, n_k, n_z)``.
        flow_wait : tf.Tensor
            Pre-computed WAIT branch flow, shape ``(n_k, n_z)``.
        k_depreciated : tf.Tensor
            Depreciated capital for each grid point, shape ``(n_k,)``.

        Returns
        -------
        v_curr : tf.Tensor
            Converged value function, ``(n_k, n_z)``.
        v_adjust_vals : tf.Tensor
            Final ADJUST branch values, ``(n_k, n_z)``.
        v_wait_vals : tf.Tensor
            Final WAIT branch values, ``(n_k, n_z)``.
        policy_k_idx : tf.Tensor
            Optimal K' indices when adjusting, ``(n_k, n_z)``.
        """
        v_curr = tf.zeros(
            (self.n_capital, self.n_productivity), dtype=TENSORFLOW_DTYPE
        )
        beta = self.params.discount_factor

        v_adjust_vals: tf.Tensor = tf.zeros_like(v_curr)
        v_wait_vals: tf.Tensor = tf.zeros_like(v_curr)
        policy_k_idx: tf.Tensor = tf.zeros_like(v_curr, dtype=tf.int32)

        for iteration in range(self.config.max_iter_vfi):
            # Expected value next period: E[V(k', z') | z]
            ev = tf.matmul(v_curr, self.P, transpose_b=True)

            # ADJUST branch: RHS(k, k', z) = flow(k, k', z) + β E[V(k', z)]
            rhs_adjust = flow_adjust + beta * tf.expand_dims(ev, 0)
            v_adjust_vals = tf.reduce_max(rhs_adjust, axis=1)
            policy_k_idx = tf.argmax(
                rhs_adjust, axis=1, output_type=tf.int32
            )

            # WAIT branch: interpolate E[V] at depreciated capital
            ev_wait = interp_1d_batch(self.k_grid, ev, k_depreciated)
            v_wait_vals = flow_wait + beta * ev_wait

            # Bellman update
            v_next = tf.maximum(v_adjust_vals, v_wait_vals)
            diff = tf.reduce_max(tf.abs(v_next - v_curr))
            v_curr = v_next

            if diff < self.config.tol_vfi:
                logger.info(
                    "VFI converged in %d iterations (diff=%.2e).",
                    iteration + 1,
                    float(diff),
                )
                break
        else:
            logger.warning(
                "VFI did not converge after %d iterations.",
                self.config.max_iter_vfi,
            )

        return v_curr, v_adjust_vals, v_wait_vals, policy_k_idx

    # ------------------------------------------------------------------
    # Result packaging
    # ------------------------------------------------------------------

    def _build_result_dict(
        self,
        v_curr: tf.Tensor,
        v_adjust_vals: tf.Tensor,
        v_wait_vals: tf.Tensor,
        policy_k_idx: tf.Tensor,
        policy_k_values: tf.Tensor,
    ) -> Dict[str, Array]:
        """Package solver outputs into a serialisable dictionary.

        All tensors are converted to NumPy arrays so that the result can
        be saved to disk without TensorFlow dependencies.

        Returns
        -------
        dict
            Keys documented in :meth:`solve`.
        """
        return {
            # Value functions
            "V": v_curr.numpy(),
            "V_adjust": v_adjust_vals.numpy(),
            "V_wait": v_wait_vals.numpy(),
            # Grids
            "K": self.k_grid.numpy(),
            "Z": self.z_grid.numpy(),
            # Policy (both discrete and continuous forms)
            "policy_adjust_idx": policy_k_idx.numpy(),
            "policy_k_values": policy_k_values.numpy(),
            # Transition matrix
            "transition_matrix": self.P.numpy(),
            # Metadata consumed by simulators
            "k_ss": float(self.k_ss),
            "k_min": self.k_min,
            "k_max": self.k_max,
            "z_min": float(self.z_min),
            "z_max": float(self.z_max),
            "depreciation_rate": float(self.params.depreciation_rate),
        }

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def solve(self) -> Dict[str, Array]:
        """Solve the basic model via value function iteration.

        Returns
        -------
        dict
            ``V``
                Converged value function, shape ``(n_k, n_z)``.
            ``V_adjust``
                ADJUST branch values, ``(n_k, n_z)``.
            ``V_wait``
                WAIT branch values, ``(n_k, n_z)``.
            ``K``
                Capital grid, ``(n_k,)``.
            ``Z``
                Productivity grid, ``(n_z,)``.
            ``policy_adjust_idx``
                Optimal K' grid index when adjusting, ``(n_k, n_z)``.
            ``policy_k_values``
                Optimal K' as continuous values, ``(n_k, n_z)``.
            ``transition_matrix``
                Productivity transition matrix, ``(n_z, n_z)``.
            ``k_ss``, ``k_min``, ``k_max``, ``z_min``, ``z_max``
                Grid metadata for simulation.
            ``depreciation_rate``
                Capital depreciation rate δ.
        """
        logger.info(
            "Starting BasicModelVFI.solve() — "
            "ψ_fixed=%.4f, ψ_convex=%.4f, n_k=%d, n_z=%d",
            self.params.adjustment_cost_fixed,
            self.params.adjustment_cost_convex,
            self.n_capital,
            self.n_productivity,
        )

        # Pre-compute flow utilities (invariant across Bellman iterations)
        flow_adjust = self.compute_flow_adjust()
        flow_wait = self.compute_flow_wait()
        k_depreciated = (1.0 - self.params.depreciation_rate) * self.k_grid

        # Bellman iteration
        v_curr, v_adjust_vals, v_wait_vals, policy_k_idx = (
            self._run_bellman_iteration(flow_adjust, flow_wait, k_depreciated)
        )

        # Log adjust-vs-wait diagnostics
        gap = v_adjust_vals - v_wait_vals
        logger.info(
            "Adjust-Wait gap: min=%.4f, max=%.4f, mean=%.4f",
            float(tf.reduce_min(gap)),
            float(tf.reduce_max(gap)),
            float(tf.reduce_mean(gap)),
        )
        logger.info(
            "Fraction adjusting: %.2f%%",
            float(tf.reduce_mean(tf.cast(gap > 0, tf.float32))) * 100,
        )

        # Policy extraction via extracted module
        policy_k_values = extract_basic_policies(self.k_grid, policy_k_idx)

        return self._build_result_dict(
            v_curr, v_adjust_vals, v_wait_vals, policy_k_idx, policy_k_values
        )
