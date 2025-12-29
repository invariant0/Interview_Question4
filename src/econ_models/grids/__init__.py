# econ_models/core/grids/__init__.py
"""
Grid generation utilities for VFI methods.

This package provides grid construction and transition matrix
generation using methods like Tauchen's approximation.
"""

from econ_models.grids.tauchen import tauchen_discretization

__all__ = ['tauchen_discretization']