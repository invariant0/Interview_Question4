# econ_models/vfi/grids/grid_utils.py
"""
Grid utility functions for VFI solvers.

All interpolation routines are XLA-compiled TensorFlow ops, providing
a unified interface for both VFI solvers and simulators.

Contains:
    * ``interp_1d_batch``  / ``_interp_1d_batch_core``  – 1-D linear (XLA)
    * ``interp_2d_batch``  / ``_interp_2d_batch_core``  – 2-D bilinear (XLA)
    * ``interp_3d_batch``  / ``_interp_3d_batch_core``  – 3-D trilinear (XLA)

``_*_core`` variants are undecorated for nesting inside other
``@tf.function(jit_compile=True)`` methods; the public wrappers
carry the XLA decorator for standalone use.

These are pure numerical routines that operate on TensorFlow tensors
and do not depend on any model-specific state beyond the grids
and pre-computed arrays passed as arguments.
"""

import logging
import math
from typing import Tuple

import tensorflow as tf
import tensorflow_probability as tfp

from econ_models.core.types import TENSORFLOW_DTYPE

ACCUM_DTYPE = TENSORFLOW_DTYPE
tfd = tfp.distributions

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
#  VRAM-aware Chunk Strategy (learned from RTX 5090 benchmarks)
# ═══════════════════════════════════════════════════════════════════════════

def compute_optimal_chunks(
    n_k: int,
    n_d: int,
    n_z: int = 15,
    vram_limit_gb: float = 28.0,
) -> Tuple[int, int, int, int]:
    """
    Return (k_chunk, b_chunk, kp_chunk, bp_chunk) with a dynamic strategy
    that avoids unnecessary chunking overhead.

    Strategy
    --------
    1. Estimate the full 5-D tile VRAM:  n_k × n_d × n_k × n_d × n_z × 4B × 3
       (×3 accounts for multiple live tensors: flow, continuation, result)
    2. If the full tile fits in VRAM (≤ ``vram_limit_gb``), set ALL chunks = full grid
       → 1 tile total, zero Python-loop overhead, maximum GPU utilisation.
    3. Otherwise, keep state dims full and derive the largest kp=bp that
       keeps per-tile VRAM within budget:
         kp = floor(sqrt(VRAM_LIMIT / (n_k × n_d × n_z × 12)))
    4. If even kp=10 doesn't fit with full state dims, also halve state dims.

    Parameters
    ----------
    n_k : int
        Number of capital grid points.
    n_d : int
        Number of debt (or second-state) grid points.
    n_z : int, default 15
        Number of productivity grid points.
    vram_limit_gb : float, default 28.0
        VRAM budget in GB (safety margin for a 32 GB card).

    Returns
    -------
    (k_chunk, b_chunk, kp_chunk, bp_chunk) : Tuple[int, int, int, int]
    """
    VRAM_LIMIT = int(vram_limit_gb * (1024 ** 3))
    BYTES_PER_ELEM = 4 * 3  # float32 × 3 concurrent tensors

    # Full-grid single-pass estimate
    full_bytes = n_k * n_d * n_k * n_d * n_z * BYTES_PER_ELEM

    if full_bytes <= VRAM_LIMIT:
        # Entire computation fits → one kernel call, no loops
        logger.info(
            f"  Chunks for ({n_k},{n_d}): FULL GRID (1 tile) — "
            f"estimated {full_bytes / 1e9:.2f} GB"
        )
        return n_k, n_d, n_k, n_d

    # Need to chunk.  Keep state dims full, derive best kp=bp from budget.
    k_chunk = n_k
    b_chunk = n_d
    per_elem = n_k * n_d * n_z * BYTES_PER_ELEM
    max_kp_sq = VRAM_LIMIT / per_elem
    kp_bp = max(int(math.isqrt(int(max_kp_sq))), 10)

    # Verify it fits; if not, halve state dims and recalculate
    tile_bytes = k_chunk * b_chunk * kp_bp * kp_bp * n_z * BYTES_PER_ELEM
    if tile_bytes > VRAM_LIMIT:
        k_chunk = max(n_k // 2, 1)
        b_chunk = max(n_d // 2, 1)
        per_elem2 = k_chunk * b_chunk * n_z * BYTES_PER_ELEM
        max_kp_sq2 = VRAM_LIMIT / per_elem2
        kp_bp = max(int(math.isqrt(int(max_kp_sq2))), 10)
        logger.info(f"  VRAM guard: halved state chunks to ({k_chunk}, {b_chunk})")

    n_tiles = (
        ((n_k + k_chunk - 1) // k_chunk)
        * ((n_d + b_chunk - 1) // b_chunk)
        * ((n_k + kp_bp - 1) // kp_bp)
        * ((n_d + kp_bp - 1) // kp_bp)
    )
    tile_gb = k_chunk * b_chunk * kp_bp * kp_bp * n_z * BYTES_PER_ELEM / 1e9
    logger.info(
        f"  Chunks for ({n_k},{n_d}): k={k_chunk}, b={b_chunk}, "
        f"kp={kp_bp}, bp={kp_bp} — {n_tiles} tiles, "
        f"~{tile_gb:.2f} GB/tile"
    )
    return k_chunk, b_chunk, kp_bp, kp_bp


# ═══════════════════════════════════════════════════════════════════════════
#  XLA-compiled 2-D Bilinear Interpolation
# ═══════════════════════════════════════════════════════════════════════════

def _interp_2d_batch_core(
    x_grid: tf.Tensor,
    y_grid: tf.Tensor,
    values: tf.Tensor,
    x_query: tf.Tensor,
    y_query: tf.Tensor
) -> tf.Tensor:
    """
    Core bilinear interpolation for 2D grids (undecorated).

    Called from inside other ``@tf.function(jit_compile=True)`` methods
    so XLA can fuse the computation without nested-compilation overhead.

    Args:
        x_grid: (N_x,) Tensor of x grid points.
        y_grid: (N_y,) Tensor of y grid points.
        values: (N_x, N_y) Tensor of values on grid.
        x_query: (Batch,) Tensor of x query points.
        y_query: (Batch,) Tensor of y query points.

    Returns:
        (Batch,) Tensor of interpolated values.
    """
    x_grid = tf.cast(x_grid, ACCUM_DTYPE)
    y_grid = tf.cast(y_grid, ACCUM_DTYPE)
    values = tf.cast(values, ACCUM_DTYPE)
    x_query = tf.cast(x_query, ACCUM_DTYPE)
    y_query = tf.cast(y_query, ACCUM_DTYPE)

    n_x = tf.shape(x_grid)[0]
    n_y = tf.shape(y_grid)[0]

    # Clamp queries to grid bounds
    x_q = tf.clip_by_value(x_query, x_grid[0], x_grid[-1])
    y_q = tf.clip_by_value(y_query, y_grid[0], y_grid[-1])

    # Find surrounding indices for x
    x_idx = tf.searchsorted(x_grid, x_q, side='right') - 1
    x_idx = tf.clip_by_value(x_idx, 0, n_x - 2)
    x_idx = tf.cast(x_idx, tf.int32)

    # Find surrounding indices for y
    y_idx = tf.searchsorted(y_grid, y_q, side='right') - 1
    y_idx = tf.clip_by_value(y_idx, 0, n_y - 2)
    y_idx = tf.cast(y_idx, tf.int32)

    # Grid values at corners
    x_lo = tf.gather(x_grid, x_idx)
    x_hi = tf.gather(x_grid, x_idx + 1)
    y_lo = tf.gather(y_grid, y_idx)
    y_hi = tf.gather(y_grid, y_idx + 1)

    # Interpolation weights
    dx = tf.maximum(x_hi - x_lo, tf.constant(1e-10, dtype=ACCUM_DTYPE))
    dy = tf.maximum(y_hi - y_lo, tf.constant(1e-10, dtype=ACCUM_DTYPE))

    wx = tf.clip_by_value((x_q - x_lo) / dx, 0.0, 1.0)
    wy = tf.clip_by_value((y_q - y_lo) / dy, 0.0, 1.0)

    # Use flat indexing instead of gather_nd for better performance
    x_idx_p1 = x_idx + 1
    y_idx_p1 = y_idx + 1

    # Compute flat indices: flat_idx = x_idx * n_y + y_idx
    values_flat = tf.reshape(values, [-1])

    flat_00 = x_idx * n_y + y_idx
    flat_01 = x_idx * n_y + y_idx_p1
    flat_10 = x_idx_p1 * n_y + y_idx
    flat_11 = x_idx_p1 * n_y + y_idx_p1

    v00 = tf.gather(values_flat, flat_00)
    v01 = tf.gather(values_flat, flat_01)
    v10 = tf.gather(values_flat, flat_10)
    v11 = tf.gather(values_flat, flat_11)

    # Bilinear interpolation - reduce arithmetic ops
    wx1 = 1.0 - wx
    wy1 = 1.0 - wy

    result = (
        v00 * (wx1 * wy1) +
        v01 * (wx1 * wy) +
        v10 * (wx * wy1) +
        v11 * (wx * wy)
    )

    return result


@tf.function(jit_compile=True)
def interp_2d_batch(
    x_grid: tf.Tensor,
    y_grid: tf.Tensor,
    values: tf.Tensor,
    x_query: tf.Tensor,
    y_query: tf.Tensor
) -> tf.Tensor:
    """
    XLA-compiled bilinear interpolation for batched queries on a 2D grid.

    Standalone wrapper for use outside other ``@tf.function`` scopes.
    Inside an XLA-compiled method, call ``_interp_2d_batch_core`` directly.

    Args:
        x_grid: (N_x,) sorted grid points along first axis.
        y_grid: (N_y,) sorted grid points along second axis.
        values: (N_x, N_y) values on grid.
        x_query: (Batch,) query points along first axis.
        y_query: (Batch,) query points along second axis.

    Returns:
        (Batch,) interpolated values.
    """
    return _interp_2d_batch_core(x_grid, y_grid, values, x_query, y_query)


# ═══════════════════════════════════════════════════════════════════════════
#  Batch-Vectorised 1-D Linear Interpolation
# ═══════════════════════════════════════════════════════════════════════════

def _interp_1d_batch_core(
    x_grid: tf.Tensor,
    y_vals: tf.Tensor,
    x_query: tf.Tensor,
) -> tf.Tensor:
    """
    Core batch 1-D linear interpolation (undecorated).

    Called from inside other @tf.function(jit_compile=True) methods so
    that XLA can fuse the gather + weighted-sum into the surrounding
    computation graph without nested-compilation overhead.

    Args:
        x_grid:  (N,)    sorted 1-D knot positions.
        y_vals:  (N, ...) values at knot positions; trailing dims are batched.
        x_query: (M,)    query positions.

    Returns:
        (M, ...) interpolated values with the same trailing shape as y_vals.
    """
    x_grid = tf.cast(x_grid, ACCUM_DTYPE)
    y_vals = tf.cast(y_vals, ACCUM_DTYPE)
    x_query = tf.cast(x_query, ACCUM_DTYPE)

    n = tf.shape(x_grid)[0]
    xq = tf.reshape(x_query, [-1])

    idx_hi = tf.searchsorted(x_grid, xq, side='right')
    idx_hi = tf.clip_by_value(idx_hi, 1, n - 1)
    idx_lo = idx_hi - 1

    x_lo = tf.gather(x_grid, idx_lo)
    x_hi = tf.gather(x_grid, idx_hi)

    denom = tf.maximum(x_hi - x_lo, tf.constant(1e-10, dtype=ACCUM_DTYPE))
    w = tf.clip_by_value((xq - x_lo) / denom, 0.0, 1.0)

    y_lo = tf.gather(y_vals, idx_lo, axis=0)
    y_hi = tf.gather(y_vals, idx_hi, axis=0)

    y_rank = len(y_vals.shape)
    if y_rank > 1:
        w = tf.reshape(w, [-1] + [1] * (y_rank - 1))

    return (1.0 - w) * y_lo + w * y_hi


@tf.function(jit_compile=True)
def interp_1d_batch(
    x_grid: tf.Tensor,
    y_vals: tf.Tensor,
    x_query: tf.Tensor,
) -> tf.Tensor:
    """
    XLA-compiled, batch-vectorised 1-D linear interpolation.

    Standalone wrapper for use outside other @tf.function scopes.
    Inside an XLA-compiled method, call ``_interp_1d_batch_core`` directly.

    Args:
        x_grid:  (N,) sorted knot positions.
        y_vals:  (N, ...) values at knot positions.
        x_query: (M,) query positions.

    Returns:
        (M, ...) interpolated values.
    """
    return _interp_1d_batch_core(x_grid, y_vals, x_query)


# ═══════════════════════════════════════════════════════════════════════════
#  XLA-compiled 3-D Trilinear Interpolation
# ═══════════════════════════════════════════════════════════════════════════

def _interp_3d_batch_core(
    k_grid: tf.Tensor,
    b_grid: tf.Tensor,
    z_grid: tf.Tensor,
    values: tf.Tensor,
    k_query: tf.Tensor,
    b_query: tf.Tensor,
    z_query: tf.Tensor,
) -> tf.Tensor:
    """
    Core trilinear interpolation for 3D grids (undecorated).

    Called from inside other ``@tf.function(jit_compile=True)`` methods
    so XLA can fuse the computation without nested-compilation overhead.

    Args:
        k_grid: (N_k,) sorted grid points along first axis.
        b_grid: (N_b,) sorted grid points along second axis.
        z_grid: (N_z,) sorted grid points along third axis.
        values: (N_k, N_b, N_z) values on grid.
        k_query: (Batch,) query points along first axis.
        b_query: (Batch,) query points along second axis.
        z_query: (Batch,) query points along third axis.

    Returns:
        (Batch,) interpolated values.
    """
    k_grid = tf.cast(k_grid, ACCUM_DTYPE)
    b_grid = tf.cast(b_grid, ACCUM_DTYPE)
    z_grid = tf.cast(z_grid, ACCUM_DTYPE)
    values = tf.cast(values, ACCUM_DTYPE)
    k_query = tf.cast(k_query, ACCUM_DTYPE)
    b_query = tf.cast(b_query, ACCUM_DTYPE)
    z_query = tf.cast(z_query, ACCUM_DTYPE)

    nk = tf.shape(k_grid)[0]
    nb = tf.shape(b_grid)[0]
    nz = tf.shape(z_grid)[0]

    # Clamp queries to grid bounds
    kq = tf.clip_by_value(k_query, k_grid[0], k_grid[-1])
    bq = tf.clip_by_value(b_query, b_grid[0], b_grid[-1])
    zq = tf.clip_by_value(z_query, z_grid[0], z_grid[-1])

    # Bracket indices via searchsorted
    ki = tf.clip_by_value(
        tf.searchsorted(k_grid, kq, side='right') - 1, 0, nk - 2)
    bi = tf.clip_by_value(
        tf.searchsorted(b_grid, bq, side='right') - 1, 0, nb - 2)
    zi = tf.clip_by_value(
        tf.searchsorted(z_grid, zq, side='right') - 1, 0, nz - 2)

    ki = tf.cast(ki, tf.int32)
    bi = tf.cast(bi, tf.int32)
    zi = tf.cast(zi, tf.int32)

    # Grid values at bracket corners
    k0 = tf.gather(k_grid, ki)
    k1 = tf.gather(k_grid, ki + 1)
    b0 = tf.gather(b_grid, bi)
    b1 = tf.gather(b_grid, bi + 1)
    z0 = tf.gather(z_grid, zi)
    z1 = tf.gather(z_grid, zi + 1)

    # Interpolation weights
    eps = tf.constant(1e-10, dtype=ACCUM_DTYPE)
    wk = (kq - k0) / tf.maximum(k1 - k0, eps)
    wb = (bq - b0) / tf.maximum(b1 - b0, eps)
    wz = (zq - z0) / tf.maximum(z1 - z0, eps)

    # Eight corner lookups via gather_nd
    ki1 = ki + 1
    bi1 = bi + 1
    zi1 = zi + 1

    def _gather(i, j, k):
        return tf.gather_nd(values, tf.stack([i, j, k], axis=1))

    v000 = _gather(ki,  bi,  zi)
    v001 = _gather(ki,  bi,  zi1)
    v010 = _gather(ki,  bi1, zi)
    v011 = _gather(ki,  bi1, zi1)
    v100 = _gather(ki1, bi,  zi)
    v101 = _gather(ki1, bi,  zi1)
    v110 = _gather(ki1, bi1, zi)
    v111 = _gather(ki1, bi1, zi1)

    # Trilinear combination
    wk1 = 1.0 - wk
    wb1 = 1.0 - wb
    wz1 = 1.0 - wz

    result = (
        v000 * (wk1 * wb1 * wz1)
        + v001 * (wk1 * wb1 * wz)
        + v010 * (wk1 * wb * wz1)
        + v011 * (wk1 * wb * wz)
        + v100 * (wk * wb1 * wz1)
        + v101 * (wk * wb1 * wz)
        + v110 * (wk * wb * wz1)
        + v111 * (wk * wb * wz)
    )
    return result


@tf.function(jit_compile=True)
def interp_3d_batch(
    k_grid: tf.Tensor,
    b_grid: tf.Tensor,
    z_grid: tf.Tensor,
    values: tf.Tensor,
    k_query: tf.Tensor,
    b_query: tf.Tensor,
    z_query: tf.Tensor,
) -> tf.Tensor:
    """
    XLA-compiled trilinear interpolation for batched queries on a 3D grid.

    Standalone wrapper for use outside other ``@tf.function`` scopes.
    Inside an XLA-compiled method, call ``_interp_3d_batch_core`` directly.

    Args:
        k_grid: (N_k,) sorted grid points along first axis.
        b_grid: (N_b,) sorted grid points along second axis.
        z_grid: (N_z,) sorted grid points along third axis.
        values: (N_k, N_b, N_z) values on grid.
        k_query: (Batch,) query points along first axis.
        b_query: (Batch,) query points along second axis.
        z_query: (Batch,) query points along third axis.

    Returns:
        (Batch,) interpolated values.
    """
    return _interp_3d_batch_core(
        k_grid, b_grid, z_grid, values, k_query, b_query, z_query,
    )


# ═══════════════════════════════════════════════════════════════════════════
#  Tauchen AR(1) Discretization
# ═══════════════════════════════════════════════════════════════════════════

def tauchen_discretization(
    n: int,
    rho: float,
    sigma: float,
    m: float = 3.0
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Discretize an AR(1) process using Tauchen's method.

    The AR(1) process: ln(z') = rho * ln(z) + epsilon
    where epsilon ~ N(0, sigma^2)

    Args:
        n: Number of grid points.
        rho: Persistence parameter of the AR(1) process.
        sigma: Standard deviation of the innovation term.
        m: Width of the grid in standard deviations.

    Returns:
        Tuple containing:
            - z: Discretized state grid (in levels, not logs).
            - p_matrix: Transition probability matrix P[i,j] = Pr(z'=j|z=i).
    """
    n_int = int(n)
    rho_t = tf.cast(rho, TENSORFLOW_DTYPE)
    sigma_t = tf.cast(sigma, TENSORFLOW_DTYPE)
    m_t = tf.cast(m, TENSORFLOW_DTYPE)
    one = tf.cast(1.0, TENSORFLOW_DTYPE)

    # Unconditional standard deviation
    std_y = sigma_t / tf.sqrt(one - rho_t ** 2)
    x_max = m_t * std_y

    # Create equally-spaced grid in log space
    x = tf.linspace(-x_max, x_max, n_int)
    x = tf.cast(x, TENSORFLOW_DTYPE)
    step = x[1] - x[0]

    # Standard normal distribution for CDF calculations
    dist = tfd.Normal(loc=tf.cast(0.0, TENSORFLOW_DTYPE), scale=one)

    # Build transition matrix using broadcasting
    p_matrix = _build_transition_matrix(x, rho_t, sigma_t, step, dist)

    # Convert from log to level
    z = tf.exp(x)

    return tf.cast(z, TENSORFLOW_DTYPE), tf.cast(p_matrix, TENSORFLOW_DTYPE)


def _build_transition_matrix(
    x: tf.Tensor,
    rho: tf.Tensor,
    sigma: tf.Tensor,
    step: tf.Tensor,
    dist: tfd.Normal
) -> tf.Tensor:
    """
    Build the Markov transition matrix for the discretized process.

    Args:
        x: Log-space grid points.
        rho: Persistence parameter.
        sigma: Innovation standard deviation.
        step: Grid spacing.
        dist: Standard normal distribution for CDF.

    Returns:
        Row-normalized transition probability matrix.
    """
    one = tf.cast(1.0, TENSORFLOW_DTYPE)

    # Broadcasting: x_j is next state, x_i is current state
    x_j = x[None, :]  # Shape: (1, n)
    x_i = x[:, None]  # Shape: (n, 1)

    # Standardized upper and lower bounds
    upper = (x_j + step / 2.0 - rho * x_i) / sigma
    lower = (x_j - step / 2.0 - rho * x_i) / sigma

    # Middle columns: probability between bounds
    p_middle = dist.cdf(upper) - dist.cdf(lower)

    # First column: cumulative up to first state + half step
    p_col0 = dist.cdf((x[0] + step / 2.0 - rho * x_i) / sigma)

    # Last column: 1 - CDF up to last state - half step
    p_coln = one - dist.cdf((x[-1] - step / 2.0 - rho * x_i) / sigma)

    # Assemble matrix
    p_matrix = tf.concat([p_col0, p_middle[:, 1:-1], p_coln], axis=1)

    # Normalize rows to ensure valid probability distribution
    row_sums = tf.reduce_sum(p_matrix, axis=1, keepdims=True)
    p_matrix = p_matrix / row_sums

    return p_matrix
