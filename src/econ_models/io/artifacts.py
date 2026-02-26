# econ_models/io/artifacts.py
"""
Utilities for saving and loading numerical artifacts.

This module handles persistence of VFI results, model weights,
and other numerical data using NumPy's compressed format.

Example:
    >>> from econ_models.io.artifacts import save_vfi_results, load_vfi_results
    >>> results = {"V": value_array, "K": capital_grid}
    >>> save_vfi_results(results, "results.npz")
"""

import numpy as np
from typing import Dict, Any

from econ_models.io.file_utils import load_json_file


def save_vfi_results(results: Dict[str, Any], filename: str) -> None:
    """
    Save VFI results to a compressed NumPy file.

    Args:
        results: Dictionary containing arrays (V, K, B, Z, Q, etc.).
        filename: Target file path (should end with .npz).
    """
    with open(filename, "wb") as f:
        np.savez(f, **results)
    print(f"Saved VFI results to {filename}")


def load_vfi_results(filename: str) -> Dict[str, np.ndarray]:
    """
    Load VFI results from a compressed NumPy file.

    Args:
        filename: Path to the .npz file.

    Returns:
        Dictionary containing loaded arrays.
    """
    with np.load(filename) as data:
        return {key: data[key] for key in data.files}