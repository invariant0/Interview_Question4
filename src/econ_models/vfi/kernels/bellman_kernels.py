"""Bellman-iteration XLA kernels shared between basic and risky models.

Contains three small XLA kernels:
- ``compute_ev`` — expected-value computation
- ``bellman_update`` — V_new = max(V_adjust, V_wait, 0) and sup-norm diff
- ``sup_norm_diff`` — ‖a − b‖∞
"""

from __future__ import annotations

from typing import Tuple

import tensorflow as tf

from econ_models.core.types import TENSORFLOW_DTYPE

ACCUM_DTYPE: tf.DType = TENSORFLOW_DTYPE


def compute_ev_core(
    v_curr: tf.Tensor,
    P: tf.Tensor,
    beta: tf.Tensor,
) -> tf.Tensor:
    """Compute discounted expected continuation value (undecorated).

    Returns ``β · V @ Pᵀ``.  Works for both 2-D ``(nk, nz)`` and
    3-D ``(nk, nb, nz)`` value functions.

    Parameters
    ----------
    v_curr : tf.Tensor
        Current value function, ``(nk, nb, nz)`` or ``(nk, nz)``.
    P : tf.Tensor
        Markov transition matrix, ``(nz, nz)``.
    beta : tf.Tensor
        Scalar discount factor.

    Returns
    -------
    tf.Tensor
        Discounted expected value, same shape as *v_curr*.
    """
    if len(v_curr.shape) == 3:
        ev = tf.tensordot(v_curr, P, axes=[[2], [1]])
    else:
        ev = tf.matmul(v_curr, P, transpose_b=True)
    return beta * ev


@tf.function(jit_compile=True)
def compute_ev(
    v_curr: tf.Tensor,
    P: tf.Tensor,
    beta: tf.Tensor,
) -> tf.Tensor:
    """Compute discounted expected continuation value (XLA-compiled).

    See :func:`compute_ev_core` for parameter documentation.
    """
    return compute_ev_core(v_curr, P, beta)


def bellman_update_core(
    v_adjust: tf.Tensor,
    v_wait: tf.Tensor,
    v_curr: tf.Tensor,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Compute ``V_new = max(V_adjust, V_wait, 0)`` and sup-norm diff (undecorated).

    Parameters
    ----------
    v_adjust, v_wait, v_curr : tf.Tensor
        All same shape in ACCUM_DTYPE.

    Returns
    -------
    v_next : tf.Tensor
        Updated value function.
    diff : tf.Tensor
        Scalar sup-norm distance ``‖V_next − V_curr‖∞``.
    """
    v_continue = tf.maximum(v_adjust, v_wait)
    v_next = tf.maximum(v_continue, tf.zeros_like(v_continue))
    diff = tf.reduce_max(tf.abs(v_next - v_curr))
    return v_next, diff


@tf.function(jit_compile=True)
def bellman_update(
    v_adjust: tf.Tensor,
    v_wait: tf.Tensor,
    v_curr: tf.Tensor,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Compute ``V_new = max(V_adjust, V_wait, 0)`` and sup-norm diff (XLA).

    See :func:`bellman_update_core` for parameter documentation.
    """
    return bellman_update_core(v_adjust, v_wait, v_curr)


def sup_norm_diff_core(a: tf.Tensor, b: tf.Tensor) -> tf.Tensor:
    """Compute the sup-norm ``‖a − b‖∞`` (undecorated)."""
    return tf.reduce_max(tf.abs(a - b))


@tf.function(jit_compile=True)
def sup_norm_diff(a: tf.Tensor, b: tf.Tensor) -> tf.Tensor:
    """Compute the sup-norm ``‖a − b‖∞`` (XLA-compiled)."""
    return sup_norm_diff_core(a, b)
