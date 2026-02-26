"""Compute tile dimensions from VRAM budget.

Re-exports ``compute_optimal_chunks`` from ``grid_utils`` for backward
compatibility and adds a clear conceptual home for tile-strategy logic.
"""

from __future__ import annotations

from econ_models.vfi.grids.grid_utils import compute_optimal_chunks

__all__ = ["compute_optimal_chunks"]
