"""Tile-index remapping and running-best accumulation kernel.

Maps local tile indices to global flat indices and tracks the running
best value and policy across tiles.  Extracted from
``RiskyDebtModelVFI._chunk_accumulate``.
"""

from __future__ import annotations

from typing import Tuple

import tensorflow as tf


def chunk_accumulate_core(
    v_chunk: tf.Tensor,
    idx_chunk: tf.Tensor,
    v_best: tf.Tensor,
    idx_best: tf.Tensor,
    kp_start_t: tf.Tensor,
    bp_start_t: tf.Tensor,
    nb_t: tf.Tensor,
    cbp_t: tf.Tensor,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Map local tile indices to global flat indices and accumulate (undecorated).

    All integer arguments are passed as scalar int32 tensors so
    that XLA traces a single generic kernel (no retracing per chunk).

    Parameters
    ----------
    v_chunk, v_best : tf.Tensor
        Per-state values for the current tile and running best.
    idx_chunk, idx_best : tf.Tensor
        Flat policy indices (local vs. global).
    kp_start_t, bp_start_t : tf.Tensor
        Offsets of the current tile in the choice grid.
    nb_t : tf.Tensor
        Total number of debt grid points (for flat-index arithmetic).
    cbp_t : tf.Tensor
        Debt-chunk size of the current tile.

    Returns
    -------
    v_best_new : tf.Tensor
        Updated running-best values.
    idx_best_new : tf.Tensor
        Updated running-best global flat indices.
    """
    local_kp = idx_chunk // cbp_t
    local_bp = idx_chunk % cbp_t
    global_flat = (
        (local_kp + kp_start_t) * nb_t
        + (local_bp + bp_start_t)
    )
    improve = v_chunk > v_best
    v_best_new = tf.where(improve, v_chunk, v_best)
    idx_best_new = tf.where(improve, global_flat, idx_best)
    return v_best_new, idx_best_new


@tf.function(jit_compile=True)
def chunk_accumulate(
    v_chunk: tf.Tensor,
    idx_chunk: tf.Tensor,
    v_best: tf.Tensor,
    idx_best: tf.Tensor,
    kp_start_t: tf.Tensor,
    bp_start_t: tf.Tensor,
    nb_t: tf.Tensor,
    cbp_t: tf.Tensor,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Map local tile indices to global flat indices and accumulate (XLA).

    See :func:`chunk_accumulate_core` for parameter documentation.
    """
    return chunk_accumulate_core(
        v_chunk, idx_chunk,
        v_best, idx_best,
        kp_start_t, bp_start_t,
        nb_t, cbp_t,
    )
