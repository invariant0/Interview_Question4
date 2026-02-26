"""Debt-inflow + tax-shield pre-computation for the ADJUST branch.

Pre-computes the debt-component tensor used by the ADJUST branch in
each outer iteration, extracted from the per-outer-iteration logic in
``RiskyDebtModelVFI.solve``.
"""

from __future__ import annotations

import tensorflow as tf

from econ_models.config.economic_params import EconomicParams
from econ_models.econ import DebtFlowCalculator


def build_debt_components(
    b_grid: tf.Tensor,
    bond_prices: tf.Tensor,
    n_debt: int,
    params: EconomicParams,
) -> tf.Tensor:
    """Pre-compute debt-inflow + tax-shield tensor.

    Parameters
    ----------
    b_grid : tf.Tensor
        ``(nb,)`` debt grid.
    bond_prices : tf.Tensor
        Current bond-price schedule, ``(nk, nb, nz)``.
    n_debt : int
        Number of debt grid points.
    params : EconomicParams
        Structural economic parameters.

    Returns
    -------
    tf.Tensor
        Compact ``(nk, nb, nz)`` debt-component tensor (debt inflow +
        tax shield).
    """
    b_next_bcast = tf.reshape(b_grid, (1, n_debt, 1))
    debt_inflow, tax_shield = DebtFlowCalculator.calculate(
        b_next_bcast, bond_prices, params,
    )
    return debt_inflow + tax_shield
