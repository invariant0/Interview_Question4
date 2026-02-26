"""XLA-compiled numerical kernels for VFI solvers.

Each module contains pure numerical functions decorated with
``@tf.function(jit_compile=True)``.  Corresponding ``_core`` variants
(undecorated) are provided for nesting inside other XLA scopes.

Modules
-------
adjust_kernels
    ADJUST branch 5-D tile computation.
wait_kernels
    WAIT branch flow computation and fused interpolation + reduction.
bellman_kernels
    Expected-value computation, Bellman update, and sup-norm.
bond_price_kernels
    Bond-price update from implied default probabilities.
chunk_accumulate
    Tile-index remapping and running-best accumulation.
"""

from econ_models.vfi.kernels.adjust_kernels import (
    adjust_tile_kernel,
    adjust_tile_kernel_core,
)
from econ_models.vfi.kernels.wait_kernels import (
    compute_flow_wait_xla,
    compute_flow_wait_xla_core,
    wait_branch_reduce,
    wait_branch_reduce_core,
)
from econ_models.vfi.kernels.bellman_kernels import (
    compute_ev,
    compute_ev_core,
    bellman_update,
    bellman_update_core,
    sup_norm_diff,
    sup_norm_diff_core,
)
from econ_models.vfi.kernels.bond_price_kernels import (
    update_bond_prices,
    update_bond_prices_core,
)
from econ_models.vfi.kernels.chunk_accumulate import (
    chunk_accumulate,
    chunk_accumulate_core,
)

__all__ = [
    "adjust_tile_kernel",
    "adjust_tile_kernel_core",
    "compute_flow_wait_xla",
    "compute_flow_wait_xla_core",
    "wait_branch_reduce",
    "wait_branch_reduce_core",
    "compute_ev",
    "compute_ev_core",
    "bellman_update",
    "bellman_update_core",
    "sup_norm_diff",
    "sup_norm_diff_core",
    "update_bond_prices",
    "update_bond_prices_core",
    "chunk_accumulate",
    "chunk_accumulate_core",
]
