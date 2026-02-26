"""Policy extraction and formatting for VFI solvers.

Contains pure functions that map discrete policy indices to continuous
values.  Extracted from ``BasicModelVFI._extract_policies`` and
``RiskyDebtModelVFI._extract_policies``.
"""

from __future__ import annotations

from typing import Dict

import tensorflow as tf

from econ_models.core.types import TENSORFLOW_DTYPE

ACCUM_DTYPE: tf.DType = TENSORFLOW_DTYPE


def extract_basic_policies(
    k_grid: tf.Tensor,
    policy_k_idx: tf.Tensor,
) -> tf.Tensor:
    """Map discrete policy indices to continuous capital values.

    Parameters
    ----------
    k_grid : tf.Tensor
        Capital grid, shape ``(n_k,)``.
    policy_k_idx : tf.Tensor
        Grid indices of optimal K', shape ``(n_k, n_z)``.

    Returns
    -------
    tf.Tensor
        Continuous K' values, shape ``(n_k, n_z)``.
    """
    return tf.gather(k_grid, policy_k_idx)


def extract_risky_policies(
    k_grid: tf.Tensor,
    b_grid: tf.Tensor,
    v_adjust_vals: tf.Tensor,
    v_wait_vals: tf.Tensor,
    policy_adjust_idx_flat: tf.Tensor,
    policy_b_wait_idx: tf.Tensor,
    n_debt: int,
    v_default_eps: float,
) -> Dict[str, tf.Tensor]:
    """Map discrete indices to continuous policy values for the risky model.

    Handles default-region masking: firms in default states get
    index ``-1`` and value ``NaN``.

    Parameters
    ----------
    k_grid : tf.Tensor
        Capital grid, ``(nk,)``.
    b_grid : tf.Tensor
        Debt grid, ``(nb,)``.
    v_adjust_vals : tf.Tensor
        ``(nk, nb, nz)`` ADJUST-branch values.
    v_wait_vals : tf.Tensor
        ``(nk, nb, nz)`` WAIT-branch values.
    policy_adjust_idx_flat : tf.Tensor
        ``(nk, nb, nz)`` flat index into ``(k' × b')`` grid.
    policy_b_wait_idx : tf.Tensor
        ``(nk, nb, nz)`` optimal next-debt index for WAIT.
    n_debt : int
        Number of debt grid points.
    v_default_eps : float
        Threshold below which a firm is considered to default.

    Returns
    -------
    dict
        Tensors keyed by policy name:
        ``policy_default``, ``policy_adjust``,
        ``policy_k_idx``, ``policy_b_idx``, ``policy_b_wait_idx``,
        ``policy_k_values``, ``policy_b_values``, ``policy_b_wait_values``.
    """
    nb = n_debt

    with tf.device('/CPU:0'):
        v_adjust_vals = tf.identity(v_adjust_vals)
        v_wait_vals = tf.identity(v_wait_vals)

        v_continue = tf.maximum(v_adjust_vals, v_wait_vals)
        policy_default = v_continue <= v_default_eps
        policy_adjust = v_adjust_vals > v_wait_vals

        policy_k_idx = policy_adjust_idx_flat // nb
        policy_b_idx = policy_adjust_idx_flat % nb

        policy_k_values = tf.gather(k_grid, policy_k_idx)
        policy_b_values = tf.gather(b_grid, policy_b_idx)
        policy_b_wait_values = tf.gather(b_grid, policy_b_wait_idx)

        term = tf.constant(-1, dtype=tf.int32)
        nan_val = tf.constant(float('nan'), dtype=ACCUM_DTYPE)

        return {
            "policy_default": policy_default,
            "policy_adjust": policy_adjust,
            "policy_k_idx": tf.where(
                policy_default, term, policy_k_idx
            ),
            "policy_b_idx": tf.where(
                policy_default, term, policy_b_idx
            ),
            "policy_b_wait_idx": tf.where(
                policy_default, term, policy_b_wait_idx
            ),
            "policy_k_values": tf.where(
                policy_default, nan_val, policy_k_values
            ),
            "policy_b_values": tf.where(
                policy_default, nan_val, policy_b_values
            ),
            "policy_b_wait_values": tf.where(
                policy_default, nan_val, policy_b_wait_values
            ),
        }
