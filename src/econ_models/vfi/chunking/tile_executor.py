"""Python tiling loop over (k, b, k', b') with kernel delegation.

Encapsulates the four-level Python for-loop that iterates over tiles,
calls the ADJUST kernel, and accumulates results.  Extracted from
``RiskyDebtModelVFI.compute_chunked_update_adjust``.

The executor receives kernel and accumulator as callables (or via
protocol), enabling substitution with mocks in unit tests.
"""

from __future__ import annotations

from typing import Callable, List, Tuple

import tensorflow as tf

from econ_models.core.types import TENSORFLOW_DTYPE

COMPUTE_DTYPE: tf.DType = tf.float32
ACCUM_DTYPE: tf.DType = TENSORFLOW_DTYPE

# Type aliases for the kernel callables
AdjustKernelFn = Callable[..., Tuple[tf.Tensor, tf.Tensor]]
ChunkAccumulateFn = Callable[..., Tuple[tf.Tensor, tf.Tensor]]


def execute_adjust_tiles(
    n_capital: int,
    n_debt: int,
    n_productivity: int,
    k_chunk_size: int,
    b_chunk_size: int,
    kp_chunk_size: int,
    bp_chunk_size: int,
    b_grid: tf.Tensor,
    flow_part_1_full: tf.Tensor,
    continuation_value: tf.Tensor,
    debt_comp_full: tf.Tensor,
    collateral_limit_5d: tf.Tensor,
    b_next_5d: tf.Tensor,
    penalty: tf.Tensor,
    enforce_constraint: bool,
    adjust_kernel: AdjustKernelFn,
    chunk_accumulate_fn: ChunkAccumulateFn,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Compute optimal ADJUST-branch value and policy via tiled execution.

    Tiles the 5-D computation over ``(k, b, k', b')`` to fit within
    VRAM limits.  Each tile is processed by *adjust_kernel*, and the
    per-tile results are accumulated with *chunk_accumulate_fn*.

    Parameters
    ----------
    n_capital, n_debt, n_productivity : int
        Grid dimensions.
    k_chunk_size, b_chunk_size, kp_chunk_size, bp_chunk_size : int
        Tile dimensions along each axis.
    b_grid : tf.Tensor
        ``(nb,)`` debt grid.
    flow_part_1_full : tf.Tensor
        ``(nk, 1, nk, 1, nz)`` profit − investment − adjustment cost.
    continuation_value : tf.Tensor
        ``(nk, nb, nz)`` discounted expected value β·E[V].
    debt_comp_full : tf.Tensor
        ``(nk, nb, nz)`` debt-component tensor.
    collateral_limit_5d : tf.Tensor
        ``(1, 1, nk, 1, 1)`` collateral constraint bound.
    b_next_5d : tf.Tensor
        ``(1, 1, 1, nb, 1)`` next-period debt values.
    penalty : tf.Tensor
        Scalar penalty for constraint violations.
    enforce_constraint : bool
        Whether to apply the collateral constraint.
    adjust_kernel : callable
        The ADJUST-branch XLA kernel.  Signature must match
        ``adjust_tile_kernel``.
    chunk_accumulate_fn : callable
        The chunk-accumulate kernel.  Signature must match
        ``chunk_accumulate``.

    Returns
    -------
    v_adjust : tf.Tensor
        ``(nk, nb, nz)`` optimal ADJUST-branch value.
    policy_idx : tf.Tensor
        ``(nk, nb, nz)`` flat index into ``(k' × b')`` grid (int32).
    """
    nk = n_capital
    nb = n_debt
    nz = n_productivity

    nb_t = tf.constant(nb, dtype=tf.int32)

    v_adjust_parts: List[tf.Tensor] = []
    policy_parts: List[tf.Tensor] = []

    for k_start in range(0, nk, k_chunk_size):
        k_end = min(k_start + k_chunk_size, nk)
        ck = k_end - k_start

        flow_part_1_k = flow_part_1_full[k_start:k_end]

        v_k_parts: List[tf.Tensor] = []
        idx_k_parts: List[tf.Tensor] = []

        for b_start in range(0, nb, b_chunk_size):
            b_end = min(b_start + b_chunk_size, nb)
            cb = b_end - b_start
            b_curr_chunk = b_grid[b_start:b_end]

            with tf.device('/CPU:0'):
                neg_inf = tf.constant(float('-inf'), dtype=ACCUM_DTYPE)
                v_best = tf.fill((ck, cb, nz), neg_inf)
                idx_best = tf.zeros((ck, cb, nz), dtype=tf.int32)

            for kp_start in range(0, nk, kp_chunk_size):
                kp_end = min(kp_start + kp_chunk_size, nk)
                ckp = kp_end - kp_start

                flow_part_1_chunk = flow_part_1_k[
                    :, :, kp_start:kp_end, :, :
                ]
                coll_chunk = collateral_limit_5d[
                    :, :, kp_start:kp_end, :, :
                ]

                for bp_start in range(0, nb, bp_chunk_size):
                    bp_end = min(bp_start + bp_chunk_size, nb)
                    cbp = bp_end - bp_start

                    debt_comp_chunk = tf.cast(
                        tf.reshape(
                            debt_comp_full[
                                kp_start:kp_end, bp_start:bp_end, :
                            ],
                            (1, 1, ckp, cbp, nz),
                        ),
                        COMPUTE_DTYPE,
                    )
                    cont_val_chunk = tf.cast(
                        tf.reshape(
                            continuation_value[
                                kp_start:kp_end, bp_start:bp_end, :
                            ],
                            (1, 1, ckp, cbp, nz),
                        ),
                        COMPUTE_DTYPE,
                    )
                    b_next_chunk = b_next_5d[
                        :, :, :, bp_start:bp_end, :
                    ]

                    v_chunk, idx_chunk = adjust_kernel(
                        flow_part_1_chunk,
                        b_curr_chunk,
                        debt_comp_chunk,
                        cont_val_chunk,
                        coll_chunk,
                        b_next_chunk,
                        penalty,
                        enforce_constraint,
                        ck,
                        cb,
                        ckp,
                        cbp,
                    )

                    kp_t = tf.constant(kp_start, dtype=tf.int32)
                    bp_t = tf.constant(bp_start, dtype=tf.int32)
                    cbp_t = tf.constant(cbp, dtype=tf.int32)
                    v_best, idx_best = chunk_accumulate_fn(
                        v_chunk, idx_chunk,
                        v_best, idx_best,
                        kp_t, bp_t, nb_t, cbp_t,
                    )

            v_k_parts.append(v_best)
            idx_k_parts.append(idx_best)

        v_adjust_parts.append(tf.concat(v_k_parts, axis=1))
        policy_parts.append(tf.concat(idx_k_parts, axis=1))

    v_adjust = tf.concat(v_adjust_parts, axis=0)
    policy_idx = tf.concat(policy_parts, axis=0)
    return v_adjust, policy_idx
