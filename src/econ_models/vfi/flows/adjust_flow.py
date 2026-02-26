"""ADJUST-branch flow_part_1 pre-computation.

Pre-computes the iteration-invariant ``flow_part_1_full`` tensor:
profit − investment − adjustment cost, for the ADJUST branch.
Extracted from ``RiskyDebtModelVFI._build_invariants``.
"""

from __future__ import annotations

import tensorflow as tf

from econ_models.config.economic_params import EconomicParams
from econ_models.econ import (
    AdjustmentCostCalculator,
    ProductionFunctions,
)


def build_adjust_flow_part1(
    k_grid: tf.Tensor,
    z_grid: tf.Tensor,
    n_capital: int,
    n_productivity: int,
    params: EconomicParams,
) -> tf.Tensor:
    """Pre-compute ADJUST branch flow_part_1 tensor.

    Computes ``Profit − Investment − AdjustmentCost`` for every
    ``(k_current, k_next, z)`` combination.

    Parameters
    ----------
    k_grid : tf.Tensor
        ``(nk,)`` capital grid.
    z_grid : tf.Tensor
        ``(nz,)`` productivity grid.
    n_capital : int
        Number of capital grid points.
    n_productivity : int
        Number of productivity grid points.
    params : EconomicParams
        Structural economic parameters.

    Returns
    -------
    tf.Tensor
        Shape ``(nk, 1, nk, 1, nz)`` — flow_part_1 for the ADJUST branch.
    """
    nk = n_capital
    nz = n_productivity

    k_curr_all = tf.reshape(k_grid, (nk, 1, 1, 1, 1))
    z_all = tf.reshape(z_grid, (1, 1, 1, 1, nz))
    k_next_all = tf.reshape(k_grid, (1, 1, nk, 1, 1))

    production = ProductionFunctions.cobb_douglas(
        k_curr_all, z_all, params,
    )
    profit = (1.0 - params.corporate_tax_rate) * production

    investment = ProductionFunctions.calculate_investment(
        k_curr_all, k_next_all, params,
    )
    adj_cost, _ = AdjustmentCostCalculator.calculate(
        investment, k_curr_all, params,
    )
    flow_part_1_full = profit - investment - adj_cost

    return flow_part_1_full
