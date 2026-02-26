"""ADJUST-branch XLA kernels as free functions.

Contains the XLA-compiled arithmetic for one ADJUST-branch VRAM tile.
Extracted from ``RiskyDebtModelVFI._adjust_chunk_kernel`` to enable
independent testing with synthetic tensors.

The kernel receives every tensor it needs as explicit arguments rather
than reading from ``self``.  XLA compilation behaviour is identical
because the ``@tf.function(jit_compile=True)`` decorator remains on
the outermost callable.
"""

from __future__ import annotations

from typing import Tuple

import tensorflow as tf

from econ_models.core.types import TENSORFLOW_DTYPE
from econ_models.econ import CashFlowCalculator, IssuanceCostCalculator
from econ_models.config.economic_params import EconomicParams

# -- Precision aliases (mirrored from the old risky.py) -------------------
COMPUTE_DTYPE: tf.DType = tf.float32
ACCUM_DTYPE: tf.DType = TENSORFLOW_DTYPE


def adjust_tile_kernel_core(
    flow_part_1: tf.Tensor,
    b_curr_chunk: tf.Tensor,
    debt_comp_5d_f16: tf.Tensor,
    cont_val_5d_f16: tf.Tensor,
    collateral_limit_5d: tf.Tensor,
    b_next_5d: tf.Tensor,
    penalty: tf.Tensor,
    enforce_constraint: bool,
    ck: int,
    cb: int,
    ckp: int,
    cbp: int,
    n_productivity: int,
    params: EconomicParams,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Mixed-precision ADJUST kernel for one VRAM tile (undecorated).

    This is the ``_core`` variant for nesting inside other XLA scopes.
    For standalone use, call :func:`adjust_tile_kernel` instead.

    Parameters
    ----------
    flow_part_1 : tf.Tensor
        ``(ck, 1, ckp, 1, nz)`` profit − investment − adjustment cost.
    b_curr_chunk : tf.Tensor
        ``(cb,)`` current-debt slice.
    debt_comp_5d_f16 : tf.Tensor
        ``(1, 1, ckp, cbp, nz)`` debt-inflow + tax-shield tile.
    cont_val_5d_f16 : tf.Tensor
        ``(1, 1, ckp, cbp, nz)`` continuation value tile.
    collateral_limit_5d : tf.Tensor
        ``(1, 1, ckp, 1, 1)`` collateral constraint bound.
    b_next_5d : tf.Tensor
        ``(1, 1, 1, cbp, 1)`` next-period debt values.
    penalty : tf.Tensor
        Scalar penalty for constraint violations.
    enforce_constraint : bool
        Whether to enforce the collateral constraint.
    ck, cb, ckp, cbp : int
        Tile dimensions.
    n_productivity : int
        Number of productivity grid points.
    params : EconomicParams
        Economic parameters (for issuance-cost and cash-flow calculations).

    Returns
    -------
    v_max : tf.Tensor
        ``(ck, cb, nz)`` best value in this tile (ACCUM_DTYPE).
    policy_idx : tf.Tensor
        ``(ck, cb, nz)`` flat index into ``ckp × cbp`` (int32).
    """
    nz = n_productivity

    fp1 = tf.cast(flow_part_1, COMPUTE_DTYPE)
    bc = tf.cast(
        tf.reshape(b_curr_chunk, (1, cb, 1, 1, 1)), COMPUTE_DTYPE
    )

    payout_f16 = fp1 - bc + debt_comp_5d_f16

    payout_acc = tf.cast(payout_f16, ACCUM_DTYPE)
    issuance_cost = IssuanceCostCalculator.calculate(payout_acc, params)
    flow_acc = CashFlowCalculator.risky_cash_flow(payout_acc, issuance_cost)
    flow_f16 = tf.cast(flow_acc, COMPUTE_DTYPE)

    if enforce_constraint:
        cl = tf.cast(collateral_limit_5d, COMPUTE_DTYPE)
        bn = tf.cast(b_next_5d, COMPUTE_DTYPE)
        pen = tf.cast(penalty, COMPUTE_DTYPE)
        flow_f16 = tf.where(bn <= cl, flow_f16, pen)

    rhs_f16 = flow_f16 + cont_val_5d_f16

    rhs = tf.cast(rhs_f16, ACCUM_DTYPE)
    rhs_flat = tf.reshape(rhs, (ck, cb, ckp * cbp, nz))

    v_max = tf.reduce_max(rhs_flat, axis=2)
    idx = tf.argmax(rhs_flat, axis=2, output_type=tf.int32)
    return v_max, idx


@tf.function(jit_compile=True)
def adjust_tile_kernel(
    flow_part_1: tf.Tensor,
    b_curr_chunk: tf.Tensor,
    debt_comp_5d_f16: tf.Tensor,
    cont_val_5d_f16: tf.Tensor,
    collateral_limit_5d: tf.Tensor,
    b_next_5d: tf.Tensor,
    penalty: tf.Tensor,
    enforce_constraint: bool,
    ck: int,
    cb: int,
    ckp: int,
    cbp: int,
    n_productivity: int,
    params: EconomicParams,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Mixed-precision ADJUST kernel for one VRAM tile (XLA-compiled).

    Standalone wrapper.  Inside another ``@tf.function(jit_compile=True)``
    scope, call :func:`adjust_tile_kernel_core` instead to avoid nested
    compilation overhead.

    See :func:`adjust_tile_kernel_core` for parameter documentation.
    """
    return adjust_tile_kernel_core(
        flow_part_1,
        b_curr_chunk,
        debt_comp_5d_f16,
        cont_val_5d_f16,
        collateral_limit_5d,
        b_next_5d,
        penalty,
        enforce_constraint,
        ck,
        cb,
        ckp,
        cbp,
        n_productivity,
        params,
    )
