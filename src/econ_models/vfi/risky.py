"""Value Function Iteration for the Risky Debt Model.

Solves a corporate-finance model with risky debt, endogenous default,
and bond-price feedback.  The firm each period may:

* **ADJUST** — pay adjustment costs, choose ``(K', B')``.
* **WAIT**   — keep capital depreciating, choose ``B'`` only.
* **DEFAULT** — exit and be replaced by a new firm.

Acceleration techniques
-----------------------
1. XLA compilation — ``@tf.function(jit_compile=True)`` on every
   numerically intensive kernel (now in ``vfi.kernels``).
2. Mixed precision — ``float16`` for the 5-D broadcast arithmetic in the
   ADJUST branch; ``float32`` for value-function storage and reductions.
3. Batch interpolation — fully vectorised, XLA-fused 1-D interpolation
   with batched gather (no Python loops over trailing dimensions).
4. Chunked 5-D tensor loops — VRAM-aware tiling over
   ``(k, b, k', b')`` so that large grids fit on consumer GPUs.

Architecture note
-----------------
This module is a thin orchestrator.  All XLA-compiled kernels live in
``vfi.kernels``, flow pre-computation in ``vfi.flows``, tiling logic in
``vfi.chunking``, and policy extraction in ``vfi.policies``.

Example::

    >>> from econ_models.vfi.risky import RiskyDebtModelVFI
    >>> solver = RiskyDebtModelVFI(params, config, k_bounds, b_bounds,
    ...                            k_chunk_size=50, b_chunk_size=50)
    >>> results = solver.solve()
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import tensorflow as tf

from econ_models.config.economic_params import EconomicParams
from econ_models.config.vfi_config import GridConfig
from econ_models.core.types import TENSORFLOW_DTYPE, Array
from econ_models.econ import CollateralCalculator
from econ_models.vfi.grids.grid_builder import GridBuilder

# Extracted modules
from econ_models.vfi.kernels.adjust_kernels import adjust_tile_kernel
from econ_models.vfi.kernels.bellman_kernels import (
    bellman_update,
    compute_ev,
    sup_norm_diff,
)
from econ_models.vfi.kernels.bond_price_kernels import update_bond_prices
from econ_models.vfi.kernels.chunk_accumulate import chunk_accumulate
from econ_models.vfi.kernels.wait_kernels import wait_branch_reduce
from econ_models.vfi.flows.adjust_flow import build_adjust_flow_part1
from econ_models.vfi.flows.debt_components import build_debt_components
from econ_models.vfi.flows.wait_flow import build_wait_flow
from econ_models.vfi.chunking.tile_executor import execute_adjust_tiles
from econ_models.vfi.policies import extract_risky_policies

logger = logging.getLogger(__name__)

# -- Precision aliases used throughout the module -------------------------
COMPUTE_DTYPE: tf.DType = tf.float32
"""Dtype for the 5-D broadcast arithmetic in the ADJUST branch."""
ACCUM_DTYPE: tf.DType = TENSORFLOW_DTYPE
"""Dtype for value-function storage and all reductions."""


# ═══════════════════════════════════════════════════════════════════════════
#  Model Class
# ═══════════════════════════════════════════════════════════════════════════

class RiskyDebtModelVFI:
    """VFI solver for the corporate-finance model with risky debt.

    The model features endogenous default, bond-price feedback, and
    collateral constraints.  The outer loop iterates on bond prices
    until the pricing schedule is consistent with the implied default
    probabilities; the inner loop runs standard Bellman iteration for a
    given price schedule.

    State space : ``(Capital K, Debt B, Productivity Z)``
    Choices     : ``(K', B')`` or DEFAULT

    Parameters
    ----------
    params : EconomicParams
        Structural economic parameters (frozen dataclass).
    config : GridConfig
        Grid sizes, tolerances, and iteration limits.
    k_bounds : tuple of (float, float)
        ``(k_min, k_max)`` bounds for the capital grid.
    b_bounds : tuple of (float, float)
        ``(b_min, b_max)`` bounds for the debt grid.
    k_chunk_size : int
        Number of current-capital grid points per VRAM tile.
    b_chunk_size : int
        Number of current-debt grid points per VRAM tile.
    kp_chunk_size : int, optional
        Number of next-capital grid points per tile.  Defaults to
        *k_chunk_size*.
    bp_chunk_size : int, optional
        Number of next-debt grid points per tile.  Defaults to
        *b_chunk_size*.

    Raises
    ------
    ValueError
        On any invalid configuration (e.g. ``n_debt < 2``).
    """

    def __init__(
        self,
        params: EconomicParams,
        config: GridConfig,
        k_bounds: Tuple[float, float],
        b_bounds: Tuple[float, float],
        k_chunk_size: int,
        b_chunk_size: int,
        kp_chunk_size: Optional[int] = None,
        bp_chunk_size: Optional[int] = None,
    ) -> None:
        self._validate_config(config, k_bounds, b_bounds)

        self.params: EconomicParams = params
        self.config: GridConfig = config
        self.custom_k_bounds: Tuple[float, float] = k_bounds
        self.custom_b_bounds: Tuple[float, float] = b_bounds

        self.k_chunk_size: int = k_chunk_size
        self.b_chunk_size: int = b_chunk_size
        self.kp_chunk_size: int = (
            kp_chunk_size if kp_chunk_size is not None else k_chunk_size
        )
        self.bp_chunk_size: int = (
            bp_chunk_size if bp_chunk_size is not None else b_chunk_size
        )

        self._initialize_grids()

    # ==================================================================
    #  Validation
    # ==================================================================

    @staticmethod
    def _validate_config(
        config: GridConfig,
        k_bounds: Tuple[float, float],
        b_bounds: Tuple[float, float],
    ) -> None:
        """Validate constructor arguments.

        Raises
        ------
        ValueError
            On any invalid parameter combination.
        """
        if config.n_debt < 2:
            raise ValueError(
                f"RiskyDebtModel requires n_debt >= 2, got {config.n_debt}."
            )
        k_lo, k_hi = k_bounds
        if k_lo >= k_hi:
            raise ValueError(
                f"k_bounds lower ({k_lo}) must be < upper ({k_hi})."
            )
        if k_lo <= 0.0:
            raise ValueError(
                f"k_bounds lower must be positive, got {k_lo}."
            )
        b_lo, b_hi = b_bounds
        if b_lo >= b_hi:
            raise ValueError(
                f"b_bounds lower ({b_lo}) must be < upper ({b_hi})."
            )

    # ==================================================================
    #  Grid initialisation
    # ==================================================================

    def _initialize_grids(self) -> None:
        """Build discretised state-space grids and pre-compute invariants.

        All grid construction runs on CPU to avoid MLIR kernel-gen PTX
        incompatibilities on newer GPU architectures (e.g. sm_120).
        The resulting tensors stay on CPU; XLA-compiled GPU kernels
        accept them via automatic host-to-device copy.
        """
        with tf.device('/CPU:0'):
            self.z_grid, self.P, self.z_min, self.z_max = (
                GridBuilder.build_productivity_grid(self.config, self.params)
            )
            self.k_grid, self.k_ss = GridBuilder.build_capital_grid(
                self.config,
                self.params,
                custom_bounds=self.custom_k_bounds,
                use_log_spacing=False,
            )
            self.b_grid = GridBuilder.build_debt_grid(
                self.config,
                self.k_ss,
                custom_bounds=self.custom_b_bounds,
            )

            shape = (
                self.config.n_capital,
                self.config.n_debt,
                self.config.n_productivity,
            )
            self.bond_prices = GridBuilder.build_initial_bond_prices(
                shape, self.params.risk_free_rate,
            )

            self.n_capital = int(tf.shape(self.k_grid)[0])
            self.n_debt = int(tf.shape(self.b_grid)[0])
            self.n_productivity = self.config.n_productivity

            self.k_min = float(self.k_grid[0])
            self.k_max = float(self.k_grid[-1])
            self.b_min = float(self.b_grid[0])
            self.b_max = float(self.b_grid[-1])

            self.k_depreciated = (1.0 - self.params.depreciation_rate) * self.k_grid

            self.k_grid = tf.cast(self.k_grid, ACCUM_DTYPE)
            self.b_grid = tf.cast(self.b_grid, ACCUM_DTYPE)
            self.z_grid = tf.cast(self.z_grid, ACCUM_DTYPE)
            self.P = tf.cast(self.P, ACCUM_DTYPE)
            self.k_depreciated = tf.cast(self.k_depreciated, ACCUM_DTYPE)
            self.bond_prices = tf.cast(self.bond_prices, ACCUM_DTYPE)

    # ==================================================================
    #  High-level methods
    # ==================================================================

    def _build_invariants(
        self,
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """Pre-compute tensors that are constant across all iterations.

        Built on CPU to avoid MLIR kernel-gen PTX issues on newer GPUs.

        Returns
        -------
        collateral_limit_5d : tf.Tensor
        b_next_5d : tf.Tensor
        penalty : tf.Tensor
        flow_part_1_full : tf.Tensor
        beta : tf.Tensor
        """
        nk = self.n_capital
        nb = self.n_debt

        with tf.device('/CPU:0'):
            beta = tf.cast(self.params.discount_factor, ACCUM_DTYPE)

            k_next_5d = tf.reshape(self.k_grid, (1, 1, nk, 1, 1))
            collateral_limit_5d = CollateralCalculator.calculate_limit(
                k_next_5d,
                self.z_min,
                self.params,
                self.params.collateral_recovery_fraction,
            )
            b_next_5d = tf.reshape(self.b_grid, (1, 1, 1, nb, 1))
            penalty = tf.cast(
                self.config.collateral_violation_penalty, ACCUM_DTYPE,
            )

            # Flow_part_1 via extracted module
            flow_part_1_full = build_adjust_flow_part1(
                self.k_grid,
                self.z_grid,
                self.n_capital,
                self.n_productivity,
                self.params,
            )

        return collateral_limit_5d, b_next_5d, penalty, flow_part_1_full, beta

    def _run_inner_vfi(
        self,
        v_curr: tf.Tensor,
        beta: tf.Tensor,
        flow_wait: tf.Tensor,
        debt_comp_full: tf.Tensor,
        collateral_limit_5d: tf.Tensor,
        b_next_5d: tf.Tensor,
        penalty: tf.Tensor,
        flow_part_1_full: tf.Tensor,
        enforce_constraint: bool,
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """Run the inner Bellman iteration for a fixed bond-price schedule.

        Parameters
        ----------
        v_curr : tf.Tensor
            Current value function guess, ``(nk, nb, nz)``.
        beta : tf.Tensor
            Scalar discount factor.
        flow_wait : tf.Tensor
            Pre-computed WAIT-branch flow, ``(nk, nb, nb, nz)``.
        debt_comp_full : tf.Tensor
            Compact debt component, ``(nk, nb, nz)``.
        collateral_limit_5d, b_next_5d, penalty, flow_part_1_full
            Invariants from :meth:`_build_invariants`.
        enforce_constraint : bool
            Whether to apply the collateral constraint.

        Returns
        -------
        v_curr : tf.Tensor
            Converged value function.
        v_adjust_vals : tf.Tensor
            Final ADJUST-branch values.
        v_wait_vals : tf.Tensor
            Final WAIT-branch values.
        policy_adjust_idx_flat : tf.Tensor
            Flat ``(k' × b')`` policy indices for ADJUST.
        policy_b_wait_idx : tf.Tensor
            Optimal next-debt indices for WAIT.
        """
        v_adjust_vals: Optional[tf.Tensor] = None
        v_wait_vals: Optional[tf.Tensor] = None
        policy_adjust_idx_flat: Optional[tf.Tensor] = None
        policy_b_wait_idx: Optional[tf.Tensor] = None

        nk = self.n_capital
        nb = self.n_debt
        nz = self.n_productivity

        # Create a closure for the adjust kernel that binds n_productivity + params
        def _adjust_kernel_bound(
            flow_part_1_chunk, b_curr_chunk,
            debt_comp_chunk, cont_val_chunk,
            coll_chunk, b_next_chunk,
            pen, enforce, ck, cb, ckp, cbp,
        ):
            return adjust_tile_kernel(
                flow_part_1_chunk, b_curr_chunk,
                debt_comp_chunk, cont_val_chunk,
                coll_chunk, b_next_chunk,
                pen, enforce, ck, cb, ckp, cbp,
                nz, self.params,
            )

        for iteration in range(self.config.max_iter_vfi):
            beta_ev = compute_ev(v_curr, self.P, beta)

            v_adjust_vals, policy_adjust_idx_flat = execute_adjust_tiles(
                n_capital=nk,
                n_debt=nb,
                n_productivity=nz,
                k_chunk_size=self.k_chunk_size,
                b_chunk_size=self.b_chunk_size,
                kp_chunk_size=self.kp_chunk_size,
                bp_chunk_size=self.bp_chunk_size,
                b_grid=self.b_grid,
                flow_part_1_full=flow_part_1_full,
                continuation_value=beta_ev,
                debt_comp_full=debt_comp_full,
                collateral_limit_5d=collateral_limit_5d,
                b_next_5d=b_next_5d,
                penalty=penalty,
                enforce_constraint=enforce_constraint,
                adjust_kernel=_adjust_kernel_bound,
                chunk_accumulate_fn=chunk_accumulate,
            )

            v_wait_vals, policy_b_wait_idx = wait_branch_reduce(
                self.k_grid, self.k_depreciated,
                flow_wait, beta_ev,
                nk, nb, nz,
            )

            v_next, diff = bellman_update(
                v_adjust_vals, v_wait_vals, v_curr,
            )
            v_curr = v_next
            diff_val = float(diff)

            if diff_val < self.config.tol_vfi:
                logger.info(
                    "  VFI converged in %d iterations (diff=%.2e).",
                    iteration + 1,
                    diff_val,
                )
                break
        else:
            logger.warning(
                "  VFI did not converge after %d iterations.",
                self.config.max_iter_vfi,
            )

        return (
            v_curr,
            v_adjust_vals,
            v_wait_vals,
            policy_adjust_idx_flat,
            policy_b_wait_idx,
        )

    def _update_prices(self, value_function: tf.Tensor) -> float:
        """Update bond prices and return the max absolute change.

        The XLA kernel returns a GPU tensor; the result is copied back to
        CPU so that subsequent debt-component computations stay on CPU.

        Parameters
        ----------
        value_function : tf.Tensor
            ``(nk, nb, nz)`` converged value function.

        Returns
        -------
        float
            Scalar max absolute change in bond prices.
        """
        q_updated, diff = update_bond_prices(
            value_function, self.bond_prices,
            self.k_grid, self.b_grid, self.z_grid, self.P,
            self.n_capital, self.n_debt, self.n_productivity,
            self.params,
            self.config.v_default_eps,
            self.config.b_eps,
            self.config.q_min,
            self.config.relax_q,
        )
        with tf.device('/CPU:0'):
            self.bond_prices = tf.identity(q_updated)
        return float(diff.numpy())

    def _build_result_dict(
        self,
        v_curr: tf.Tensor,
        v_adjust_vals: tf.Tensor,
        v_wait_vals: tf.Tensor,
        policies: Dict[str, tf.Tensor],
    ) -> Dict[str, Array]:
        """Package solver outputs into a serialisable dictionary.

        All tensors are converted to NumPy arrays.

        Returns
        -------
        dict
            Keys documented in :meth:`solve`.
        """
        with tf.device('/CPU:0'):
            v_curr = tf.identity(v_curr)

        return {
            "V": v_curr.numpy(),
            "V_adjust": v_adjust_vals.numpy(),
            "V_wait": v_wait_vals.numpy(),
            "Q": self.bond_prices.numpy(),
            "K": self.k_grid.numpy(),
            "B": self.b_grid.numpy(),
            "Z": self.z_grid.numpy(),
            "transition_matrix": self.P.numpy(),
            # Solver config (needed by simulator for value-based decisions)
            "v_default_eps": float(self.config.v_default_eps),
            "depreciation_rate": float(self.params.depreciation_rate),
            # Discrete policies (grid-snapped)
            "policy_default": policies["policy_default"].numpy(),
            "policy_adjust": policies["policy_adjust"].numpy(),
            "policy_k_idx": policies["policy_k_idx"].numpy(),
            "policy_b_idx": policies["policy_b_idx"].numpy(),
            "policy_b_wait_idx": policies["policy_b_wait_idx"].numpy(),
            "policy_k_values": policies["policy_k_values"].numpy(),
            "policy_b_values": policies["policy_b_values"].numpy(),
            "policy_b_wait_values": policies["policy_b_wait_values"].numpy(),
        }

    # ==================================================================
    #  Solve — main entry point
    # ==================================================================

    def solve(self) -> Dict[str, Array]:
        """Solve the full equilibrium by iterating on bond prices and values.

        The outer loop updates bond prices until the pricing schedule is
        consistent with implied default probabilities.  The inner loop
        runs Bellman iteration for a fixed price schedule.

        Returns
        -------
        dict
            ``V``
                Converged value function, ``(nk, nb, nz)``.
            ``V_adjust``, ``V_wait``
                Branch values, ``(nk, nb, nz)``.
            ``Q``
                Equilibrium bond-price schedule, ``(nk, nb, nz)``.
            ``K``, ``B``, ``Z``
                Grids.
            ``transition_matrix``
                Productivity transition matrix.
            ``policy_default``
                Boolean default indicator, ``(nk, nb, nz)``.
            ``policy_adjust``
                Boolean ADJUST indicator, ``(nk, nb, nz)``.
            ``policy_k_idx``, ``policy_b_idx``
                Discrete ADJUST policies (−1 in default region).
            ``policy_b_wait_idx``
                Discrete WAIT debt policy (−1 in default region).
            ``policy_k_values``, ``policy_b_values``,
            ``policy_b_wait_values``
                Continuous policy values (NaN in default region).
            ``v_default_eps``, ``depreciation_rate``
                Scalar metadata.
        """
        nk = self.n_capital
        nb = self.n_debt
        nz = self.n_productivity

        logger.info("=" * 64)
        logger.info("  RISKY DEBT VFI SOLVE")
        logger.info("  XLA + Mixed Precision + Batch Interpolation")
        logger.info("=" * 64)
        logger.info(
            "  Grid       : nk=%d, nb=%d, nz=%d", nk, nb, nz
        )
        logger.info(
            "  Chunks     : k=%d, b=%d, k'=%d, b'=%d",
            self.k_chunk_size,
            self.b_chunk_size,
            self.kp_chunk_size,
            self.bp_chunk_size,
        )
        logger.info("  Compute    : %s", COMPUTE_DTYPE.name)
        logger.info("  Accumulate : %s", ACCUM_DTYPE.name)
        logger.info("=" * 64)

        # Pre-compute iteration-invariant tensors
        (
            collateral_limit_5d,
            b_next_5d,
            penalty,
            flow_part_1_full,
            beta,
        ) = self._build_invariants()

        v_curr = tf.zeros((nk, nb, nz), dtype=ACCUM_DTYPE)

        # Mutable state updated per outer iteration
        policy_adjust_idx_flat: Optional[tf.Tensor] = None
        policy_b_wait_idx: Optional[tf.Tensor] = None
        v_adjust_vals: Optional[tf.Tensor] = None
        v_wait_vals: Optional[tf.Tensor] = None
        flow_wait: Optional[tf.Tensor] = None

        for outer_iter in range(self.config.max_outer):
            is_first_outer = (outer_iter == 0)
            v_old = v_curr

            # ── Free large GPU tensors from previous iteration ──────
            # flow_wait is (nk, nb, nb, nz) ≈ 7.85 GiB at 560³×12.
            # Python evaluates the RHS of an assignment before rebinding
            # the variable, so the old tensor stays alive during the
            # build_wait_flow() call.  Explicitly deleting first avoids
            # two copies coexisting and prevents BFC-allocator
            # fragmentation from causing an OOM.
            del flow_wait
            flow_wait = None

            # Debt components via extracted module (CPU)
            with tf.device('/CPU:0'):
                debt_comp_full = build_debt_components(
                    self.b_grid, self.bond_prices, nb, self.params,
                )

            # WAIT flow via extracted module
            flow_wait = build_wait_flow(
                self.k_grid, self.b_grid, self.z_grid,
                self.k_depreciated, self.bond_prices,
                nk, nb, nz, self.z_min, self.params,
                self.config.collateral_violation_penalty,
                self.params.collateral_recovery_fraction,
                enforce_constraint=is_first_outer,
            )

            # Inner Bellman iteration
            (
                v_curr,
                v_adjust_vals,
                v_wait_vals,
                policy_adjust_idx_flat,
                policy_b_wait_idx,
            ) = self._run_inner_vfi(
                v_curr,
                beta,
                flow_wait,
                debt_comp_full,
                collateral_limit_5d,
                b_next_5d,
                penalty,
                flow_part_1_full,
                enforce_constraint=is_first_outer,
            )

            err_q = self._update_prices(v_curr)
            err_v = float(sup_norm_diff(v_curr, v_old).numpy())
            logger.info(
                "  Outer %3d:  Err_V = %.6f  Err_Q = %.6f",
                outer_iter,
                err_v,
                err_q,
            )

            if err_v < self.config.tol_outer and outer_iter >= 2:
                logger.info(
                    "  Converged after %d outer iterations.",
                    outer_iter + 1,
                )
                break

        # Policy extraction via extracted module
        policies = extract_risky_policies(
            self.k_grid,
            self.b_grid,
            v_adjust_vals,
            v_wait_vals,
            policy_adjust_idx_flat,
            policy_b_wait_idx,
            nb,
            self.config.v_default_eps,
        )

        logger.info("=" * 64)

        return self._build_result_dict(
            v_curr, v_adjust_vals, v_wait_vals, policies
        )
