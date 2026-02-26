"""VRAM-aware tiling strategy and execution for the ADJUST branch.

Modules
-------
tile_strategy
    Compute tile dimensions from VRAM budget.
tile_executor
    Python tiling loop over (k, b, k', b') with kernel delegation.
"""

from econ_models.vfi.chunking.tile_strategy import compute_optimal_chunks
from econ_models.vfi.chunking.tile_executor import execute_adjust_tiles

__all__ = [
    "compute_optimal_chunks",
    "execute_adjust_tiles",
]
