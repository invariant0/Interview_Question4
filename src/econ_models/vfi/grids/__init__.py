# econ_models/vfi/grids/__init__.py
"""
Grid management for VFI models.

This package provides utilities for constructing and managing
discretized state space grids, as well as interpolation and
policy-refinement helpers used by VFI solvers.
"""

from econ_models.vfi.grids.grid_builder import GridBuilder
from econ_models.vfi.grids.grid_utils import (
    # VRAM-aware chunk strategy
    compute_optimal_chunks,
    # 1-D linear interpolation (XLA)
    _interp_1d_batch_core,
    interp_1d_batch,
    # 2-D bilinear interpolation (XLA)
    _interp_2d_batch_core,
    interp_2d_batch,
    # 3-D trilinear interpolation (XLA)
    _interp_3d_batch_core,
    interp_3d_batch,
)

__all__ = [
    'GridBuilder',
    # VRAM-aware chunk strategy
    'compute_optimal_chunks',
    # 1-D linear interpolation (XLA)
    '_interp_1d_batch_core',
    'interp_1d_batch',
    # 2-D bilinear interpolation (XLA)
    '_interp_2d_batch_core',
    'interp_2d_batch',
    # 3-D trilinear interpolation (XLA)
    '_interp_3d_batch_core',
    'interp_3d_batch',
]