"""Value Function Iteration (VFI) solvers for economic models.

This package provides:

* :class:`BasicModelVFI` — VFI solver for the basic RBC model with
  adjustment costs (2-D state space: capital × productivity).
* :class:`RiskyDebtModelVFI` — VFI solver for the risky-debt model
  with endogenous default (3-D state space: capital × debt × productivity).
* :class:`VFIEngine` — Generic Bellman fixed-point iterator.
* :class:`BoundaryFinder` — Automated grid-boundary discovery via
  solve-simulate loops.

Sub-packages
------------
kernels
    XLA-compiled numerical kernels (adjust, wait, bellman, bond-price,
    chunk-accumulate).
flows
    Flow-tensor construction (adjust, wait, debt components).
chunking
    VRAM-aware tiling strategy and tile executor.
simulation
    Post-solve simulators (basic, risky).
grids
    Grid construction and interpolation utilities.

Modules
-------
protocols
    Protocol definitions for solver components (for dependency injection
    and testability).
policies
    Policy extraction and formatting.
engine
    Generic Bellman fixed-point iterator.
"""

from econ_models.vfi.basic import BasicModelVFI
from econ_models.vfi.bounds import BoundaryFinder
from econ_models.vfi.engine import VFIEngine
from econ_models.vfi.risky import RiskyDebtModelVFI

__all__ = [
    "BasicModelVFI",
    "BoundaryFinder",
    "RiskyDebtModelVFI",
    "VFIEngine",
]