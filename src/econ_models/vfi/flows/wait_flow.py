"""WAIT-branch flow construction with bond-price interpolation.

Wraps the call to the WAIT kernel, handling the bond-price
interpolation at depreciated capital that must happen before
the kernel is called.
"""

from __future__ import annotations

import tensorflow as tf

from econ_models.config.economic_params import EconomicParams
from econ_models.vfi.grids.grid_utils import interp_1d_batch
from econ_models.vfi.kernels.wait_kernels import compute_flow_wait_xla


def build_wait_flow(
    k_grid: tf.Tensor,
    b_grid: tf.Tensor,
    z_grid: tf.Tensor,
    k_depreciated: tf.Tensor,
    bond_prices: tf.Tensor,
    n_capital: int,
    n_debt: int,
    n_productivity: int,
    z_min: float,
    params: EconomicParams,
    collateral_violation_penalty: float,
    collateral_recovery_fraction: float,
    enforce_constraint: bool,
) -> tf.Tensor:
    """Build the WAIT-branch flow tensor.

    Interpolates bond prices at depreciated capital and delegates to
    the XLA-compiled WAIT kernel.

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
    bond_prices : tf.Tensor
        Current bond-price schedule, ``(nk, nb, nz)``.
    n_capital, n_debt, n_productivity : int
        Grid dimensions.
    z_min : float
        Minimum productivity value.
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
        ``(nk, nb_curr, nb_next, nz)`` flow tensor.
    """
    q_wait = interp_1d_batch(k_grid, bond_prices, k_depreciated)
    return compute_flow_wait_xla(
        k_grid, b_grid, z_grid, k_depreciated, q_wait,
        n_capital, n_debt, n_productivity, z_min,
        params, collateral_violation_penalty,
        collateral_recovery_fraction, enforce_constraint,
    )
