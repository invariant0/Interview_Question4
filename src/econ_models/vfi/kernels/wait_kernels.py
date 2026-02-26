"""WAIT-branch XLA kernels as free functions.

Contains the XLA-compiled WAIT-branch flow computation and
the fused interpolation-plus-reduction, both extracted from
``RiskyDebtModelVFI`` to enable independent testing.

Each kernel receives all grid tensors, parameter scalars, and input
arrays as explicit arguments.
"""

from __future__ import annotations

from typing import Tuple

import tensorflow as tf

from econ_models.core.types import TENSORFLOW_DTYPE
from econ_models.config.economic_params import EconomicParams
from econ_models.econ import (
    CashFlowCalculator,
    CollateralCalculator,
    DebtFlowCalculator,
    IssuanceCostCalculator,
    ProductionFunctions,
)
from econ_models.vfi.grids.grid_utils import _interp_1d_batch_core

ACCUM_DTYPE: tf.DType = TENSORFLOW_DTYPE


def compute_flow_wait_xla_core(
    k_grid: tf.Tensor,
    b_grid: tf.Tensor,
    z_grid: tf.Tensor,
    k_depreciated: tf.Tensor,
    q_wait: tf.Tensor,
    n_capital: int,
    n_debt: int,
    n_productivity: int,
    z_min: float,
    params: EconomicParams,
    collateral_violation_penalty: float,
    collateral_recovery_fraction: float,
    enforce_constraint: bool,
) -> tf.Tensor:
    """Compute the WAIT-branch flow tensor (undecorated core).

    Parameters
    ----------
    k_grid : tf.Tensor
        ``(nk,)`` capital grid.
    b_grid : tf.Tensor
        ``(nb,)`` debt grid.
    z_grid : tf.Tensor
        ``(nz,)`` productivity grid.
    k_depreciated : tf.Tensor
        ``(nk,)`` depreciated capital values.
    q_wait : tf.Tensor
        Bond prices interpolated at depreciated capital, ``(nk, nb, nz)``.
    n_capital, n_debt, n_productivity : int
        Grid dimensions.
    z_min : float
        Minimum productivity value (for collateral calculation).
    params : EconomicParams
        Structural economic parameters.
    collateral_violation_penalty : float
        Penalty value for constraint violations.
    collateral_recovery_fraction : float
        Collateral recovery fraction parameter.
    enforce_constraint : bool
        Whether to apply the collateral constraint.

    Returns
    -------
    tf.Tensor
        Flow tensor ``(nk, nb_curr, nb_next, nz)`` in ACCUM_DTYPE.
    """
    nk = n_capital
    nb = n_debt
    nz = n_productivity

    k_4d = tf.reshape(k_grid, (nk, 1, 1, 1))
    bc_4d = tf.reshape(b_grid, (1, nb, 1, 1))
    bn_4d = tf.reshape(b_grid, (1, 1, nb, 1))
    z_4d = tf.reshape(z_grid, (1, 1, 1, nz))

    production = ProductionFunctions.cobb_douglas(k_4d, z_4d, params)
    profit = (1.0 - params.corporate_tax_rate) * production

    q_wait_4d = tf.reshape(q_wait, (nk, 1, nb, nz))
    debt_inflow, tax_shield = DebtFlowCalculator.calculate(
        bn_4d, q_wait_4d, params,
    )

    payout = profit + debt_inflow + tax_shield - bc_4d
    issuance_cost = IssuanceCostCalculator.calculate(payout, params)
    flow = CashFlowCalculator.risky_cash_flow(payout, issuance_cost)

    if enforce_constraint:
        k_dep_4d = tf.reshape(k_depreciated, (nk, 1, 1, 1))
        coll_lim = CollateralCalculator.calculate_limit(
            k_dep_4d,
            z_min,
            params,
            collateral_recovery_fraction,
        )
        penalty = tf.cast(collateral_violation_penalty, ACCUM_DTYPE)
        flow = tf.where(bn_4d <= coll_lim, flow, penalty)

    return flow


@tf.function(jit_compile=True)
def compute_flow_wait_xla(
    k_grid: tf.Tensor,
    b_grid: tf.Tensor,
    z_grid: tf.Tensor,
    k_depreciated: tf.Tensor,
    q_wait: tf.Tensor,
    n_capital: int,
    n_debt: int,
    n_productivity: int,
    z_min: float,
    params: EconomicParams,
    collateral_violation_penalty: float,
    collateral_recovery_fraction: float,
    enforce_constraint: bool,
) -> tf.Tensor:
    """Compute the WAIT-branch flow tensor (XLA-compiled).

    See :func:`compute_flow_wait_xla_core` for parameter documentation.
    """
    return compute_flow_wait_xla_core(
        k_grid, b_grid, z_grid, k_depreciated, q_wait,
        n_capital, n_debt, n_productivity, z_min,
        params, collateral_violation_penalty,
        collateral_recovery_fraction, enforce_constraint,
    )


def wait_branch_reduce_core(
    k_grid: tf.Tensor,
    k_depreciated: tf.Tensor,
    flow_wait: tf.Tensor,
    beta_ev: tf.Tensor,
    n_capital: int,
    n_debt: int,
    n_productivity: int,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Interpolate continuation value at depreciated K and reduce (undecorated).

    Fused kernel that batch-interpolates ``β · E[V]`` at depreciated
    capital, adds the WAIT-branch flow, and reduces over the
    next-period debt dimension.

    Parameters
    ----------
    k_grid : tf.Tensor
        ``(nk,)`` capital grid.
    k_depreciated : tf.Tensor
        ``(nk,)`` depreciated capital values.
    flow_wait : tf.Tensor
        ``(nk, nb_curr, nb_next, nz)`` WAIT-branch flow.
    beta_ev : tf.Tensor
        ``(nk, nb, nz)`` discounted expected value.
    n_capital, n_debt, n_productivity : int
        Grid dimensions.

    Returns
    -------
    v_wait : tf.Tensor
        ``(nk, nb, nz)`` optimal WAIT-branch value (ACCUM_DTYPE).
    policy_b_wait : tf.Tensor
        ``(nk, nb, nz)`` optimal next-period debt index (int32).
    """
    nk = n_capital
    nb = n_debt
    nz = n_productivity

    ev_wait = _interp_1d_batch_core(k_grid, beta_ev, k_depreciated)

    ev_exp = tf.reshape(ev_wait, (nk, 1, nb, nz))
    rhs = flow_wait + ev_exp

    v_wait = tf.reduce_max(rhs, axis=2)
    idx = tf.argmax(rhs, axis=2, output_type=tf.int32)
    return v_wait, idx


@tf.function(jit_compile=True)
def wait_branch_reduce(
    k_grid: tf.Tensor,
    k_depreciated: tf.Tensor,
    flow_wait: tf.Tensor,
    beta_ev: tf.Tensor,
    n_capital: int,
    n_debt: int,
    n_productivity: int,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Interpolate continuation value at depreciated K and reduce (XLA).

    See :func:`wait_branch_reduce_core` for parameter documentation.
    """
    return wait_branch_reduce_core(
        k_grid, k_depreciated, flow_wait, beta_ev,
        n_capital, n_debt, n_productivity,
    )
