"""Bond-price update XLA kernel as free function.

Contains the XLA kernel for updating bond prices from implied default
probabilities, extracted from ``RiskyDebtModelVFI._update_prices_core``.
"""

from __future__ import annotations

from typing import Tuple

import tensorflow as tf

from econ_models.core.types import TENSORFLOW_DTYPE
from econ_models.config.economic_params import EconomicParams
from econ_models.econ import BondPricingCalculator, ProductionFunctions

ACCUM_DTYPE: tf.DType = TENSORFLOW_DTYPE


def update_bond_prices_core(
    value_function: tf.Tensor,
    bond_prices_old: tf.Tensor,
    k_grid: tf.Tensor,
    b_grid: tf.Tensor,
    z_grid: tf.Tensor,
    P: tf.Tensor,
    n_capital: int,
    n_debt: int,
    n_productivity: int,
    params: EconomicParams,
    v_default_eps: float,
    b_eps: float,
    q_min: float,
    relax_q: float,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Update bond prices from implied default probabilities (undecorated).

    Parameters
    ----------
    value_function : tf.Tensor
        ``(nk, nb, nz)`` converged value function.
    bond_prices_old : tf.Tensor
        ``(nk, nb, nz)`` previous bond-price schedule.
    k_grid : tf.Tensor
        ``(nk,)`` capital grid.
    b_grid : tf.Tensor
        ``(nb,)`` debt grid.
    z_grid : tf.Tensor
        ``(nz,)`` productivity grid.
    P : tf.Tensor
        ``(nz, nz)`` Markov transition matrix.
    n_capital, n_debt, n_productivity : int
        Grid dimensions.
    params : EconomicParams
        Structural economic parameters.
    v_default_eps : float
        Threshold below which a firm is considered to default.
    b_eps : float
        Epsilon for near-zero debt in pricing formula.
    q_min : float
        Minimum bond price floor.
    relax_q : float
        Relaxation weight for blending old and new prices.

    Returns
    -------
    q_updated : tf.Tensor
        Relaxed new bond prices, ``(nk, nb, nz)``.
    diff : tf.Tensor
        Scalar max absolute change in bond prices.
    """
    nk = n_capital
    nb = n_debt
    nz = n_productivity

    is_default = tf.cast(
        value_function <= v_default_eps, ACCUM_DTYPE,
    )

    k_col = tf.reshape(k_grid, (nk, 1, 1))
    z_col = tf.reshape(z_grid, (1, 1, nz))

    profit_next = (
        (1.0 - params.corporate_tax_rate)
        * ProductionFunctions.cobb_douglas(k_col, z_col, params)
    )
    recovery = BondPricingCalculator.recovery_value(
        profit_next, k_col, params,
    )

    b_col = tf.reshape(b_grid, (1, nb, 1))
    payoff = BondPricingCalculator.bond_payoff(
        recovery, b_col, is_default,
    )

    payoff_flat = tf.reshape(payoff, (-1, nz))
    exp_flat = tf.matmul(payoff_flat, P, transpose_b=True)
    exp_payoff = tf.reshape(exp_flat, (nk, nb, nz))

    r_rf = params.risk_free_rate
    q_rf = 1.0 / (1.0 + r_rf)

    q_new = BondPricingCalculator.risk_neutral_price(
        expected_payoff=exp_payoff,
        debt_next=b_col,
        risk_free_rate=r_rf,
        epsilon_debt=b_eps,
        min_price=q_min,
        risk_free_price_val=q_rf,
    )

    lam = tf.cast(relax_q, ACCUM_DTYPE)
    q_updated = lam * q_new + (1.0 - lam) * bond_prices_old
    diff = tf.reduce_max(tf.abs(q_new - bond_prices_old))
    return q_updated, diff


@tf.function(jit_compile=True)
def update_bond_prices(
    value_function: tf.Tensor,
    bond_prices_old: tf.Tensor,
    k_grid: tf.Tensor,
    b_grid: tf.Tensor,
    z_grid: tf.Tensor,
    P: tf.Tensor,
    n_capital: int,
    n_debt: int,
    n_productivity: int,
    params: EconomicParams,
    v_default_eps: float,
    b_eps: float,
    q_min: float,
    relax_q: float,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Update bond prices from implied default probabilities (XLA-compiled).

    See :func:`update_bond_prices_core` for parameter documentation.
    """
    return update_bond_prices_core(
        value_function, bond_prices_old,
        k_grid, b_grid, z_grid, P,
        n_capital, n_debt, n_productivity,
        params, v_default_eps, b_eps, q_min, relax_q,
    )
