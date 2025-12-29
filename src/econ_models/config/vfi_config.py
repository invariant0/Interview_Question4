# econ_models/config/vfi_config.py
"""
Configuration for Value Function Iteration (VFI) solvers.

This module provides configuration classes and utilities for discrete-grid
VFI methods, including grid specifications and numerical tolerances.

Example:
    >>> from econ_models.config.vfi_config import load_grid_config
    >>> config = load_grid_config("config/vfi.json", "basic")
    >>> print(f"Capital grid points: {config.n_capital}")
"""

from dataclasses import dataclass, fields
from typing import Tuple
import os
import sys
import logging

from econ_models.core.types import TENSORFLOW_DTYPE, Tensor
from econ_models.io.file_utils import load_json_file

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GridConfig:
    """
    Configuration for VFI computational grids and numerical tolerances.

    This immutable configuration controls the discretization of state spaces
    and the convergence criteria for iterative algorithms.

    Attributes:
        n_capital: Number of points in the capital grid.
        n_productivity: Number of points in the productivity grid.
        n_debt: Number of points in the debt grid (0 for basic model).
        tauchen_width: Width in standard deviations for Tauchen's method.
        tol_vfi: Convergence tolerance for inner VFI loop.
        max_iter_vfi: Maximum iterations for inner VFI loop.
        tol_outer: Convergence tolerance for outer pricing loop.
        max_outer: Maximum iterations for outer pricing loop.
        v_default_eps: Value threshold for default detection.
        b_eps: Numerical epsilon for debt calculations.
        q_min: Minimum bond price floor.
        relax_q: Relaxation parameter for bond price updates.
        collateral_violation_penalty: Penalty for constraint violations.
    """

    n_capital: int = 100
    n_productivity: int = 7
    n_debt: int = 0
    tauchen_width: float = 3.0

    tol_vfi: float = 1e-7
    max_iter_vfi: int = 2000
    tol_outer: float = 1e-4
    max_outer: int = 200

    v_default_eps: float = 1e-10
    b_eps: float = 1e-10
    q_min: float = 1e-10
    relax_q: float = 0.5

    collateral_violation_penalty: float = -1e10


def load_grid_config(filename: str, model_type: str) -> GridConfig:
    """
    Load grid configuration from a JSON file for a specific model type.

    Args:
        filename: Path to the JSON configuration file.
        model_type: Key in the JSON file ('basic' or 'risky').

    Returns:
        Populated GridConfig instance.
    """
    if not os.path.exists(filename):
        logger.warning(
            f"Grid config file '{filename}' not found. Using defaults."
        )
        return GridConfig()

    try:
        full_data = load_json_file(filename)

        if model_type not in full_data:
            logger.warning(
                f"Key '{model_type}' not in {filename}. Using defaults."
            )
            return GridConfig()

        model_data = full_data[model_type]
        valid_keys = {f.name for f in fields(GridConfig)}
        filtered_data = {k: v for k, v in model_data.items() if k in valid_keys}

        return GridConfig(**filtered_data)

    except Exception as e:
        logger.error(f"Error reading grid config {filename}: {e}")
        sys.exit(1)